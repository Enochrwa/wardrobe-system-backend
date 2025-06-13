from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from .. import model as models, tables as schemas
from sqlalchemy import func, or_

# Added imports for AI-powered recommendations
import numpy as np # For random embeddings if needed
import random
from itertools import combinations
from sqlalchemy.orm import joinedload # For eager loading of outfit items

from ..services.outfit_matching_service import OutfitMatchingService
# from ..services.ai_services import get_fashion_trends_service # Keep commented if not fully implementing trend integration yet
# from ..services.ai_services import analyze_outfit_image_service # Not directly used if AI features are mocked/pre-stored

# For sentence embeddings for occasion matching
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except ImportError:
    import logging # Import here if not at top level already
    logger = logging.getLogger(__name__)
    logger.warning("sentence_transformers not available. Occasion matching will use fallback logic.")
    sentence_model = None
    cosine_similarity = None


# --- Load Models ---
# Models are loaded in the import section above

# Instantiate services needed
outfit_matcher = OutfitMatchingService()

from .weather_service import get_weather_data # For weather-based recommendations
import asyncio # For running async weather_service call if needed, or make the main function async

# Helper function to find matching outfits based on simple criteria (OLD - to be replaced)
# New AI-driven helper function for occasion matching
def find_ai_matched_outfits_for_occasion(
    db: Session,
    user_id: int,
    occasion_text: str, # Combined occasion name and notes
    num_recommendations: int = 3,
    min_coherence_score: float = 0.4 # Minimum internal coherence for an outfit to be considered
) -> List[models.Outfit]:
    logger = logging.getLogger(__name__) # Ensure logger is available

    if not sentence_model:
        logger.warning("Sentence transformer model not loaded. Cannot perform AI matching.")
        return []

    try:
        occasion_embedding = sentence_model.encode(occasion_text)
    except Exception as e:
        logger.error(f"Error encoding occasion text: {e}")
        return []

    # Fetch user's outfits with their items eagerly loaded
    # Assuming 'items_association' is the relationship from Outfit to OutfitItem (association object)
    # and 'item' is the relationship from OutfitItem to WardrobeItem.
    user_outfits = db.query(models.Outfit)\
        .filter(models.Outfit.user_id == user_id)\
        .options(joinedload(models.Outfit.items_association).joinedload(models.OutfitItem.item))\
        .all()

    if not user_outfits:
        return []

    scored_outfits = []

    for outfit in user_outfits:
        if not outfit.items_association: # Skip outfits with no items
            continue

        outfit_item_features_for_matcher: List[Dict[str, Any]] = []
        item_embeddings_for_outfit_avg: List[np.ndarray] = []

        for assoc_obj in outfit.items_association:
            item = assoc_obj.item # This is the WardrobeItem model
            if not item: continue

            # Use actual AI embedding if available, otherwise mock (embedding dim for all-MiniLM-L6-v2 is 384)
            item_emb_list = item.ai_embedding
            if item_emb_list is None:
                item_emb_list = np.random.rand(384).tolist()

            # Use actual AI dominant colors if available, otherwise mock
            item_colors = item.ai_dominant_colors
            if item_colors is None:
                item_colors = random.sample(["#1A1A1A", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF"], k=min(2, len(["#1A1A1A", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF"])))


            item_category = getattr(item, "category", None)
            if item_category is None:
                item_category = random.choice(["Tops", "Bottoms", "Shoes", "Accessories"])

            outfit_item_features_for_matcher.append({
                "id": item.id,
                "name": item.name,
                "embedding": item_emb_list, # Used by outfit_matcher
                "colors": item_colors,    # Used by outfit_matcher
                "category": item_category # Used by outfit_matcher (potentially)
            })
            item_embeddings_for_outfit_avg.append(np.array(item_emb_list))

        if not outfit_item_features_for_matcher or not item_embeddings_for_outfit_avg:
            continue # Not enough item data to process this outfit

        # 1. Calculate Outfit Coherence
        coherence_details = outfit_matcher.calculate_compatibility_score(outfit_item_features_for_matcher)
        internal_coherence_score = coherence_details["score"]

        if internal_coherence_score < min_coherence_score:
            continue # Skip outfits that are not internally coherent

        # 2. Create Outfit Embedding (average of item embeddings)
        outfit_embedding_avg = np.mean(item_embeddings_for_outfit_avg, axis=0)

        # 3. Calculate Match Score (Similarity to Occasion + Internal Coherence)
        similarity_to_occasion = cosine_similarity(occasion_embedding.reshape(1, -1), outfit_embedding_avg.reshape(1, -1))[0][0]
        # Normalize cosine similarity from [-1, 1] to [0, 1] for scoring
        similarity_to_occasion_normalized = (similarity_to_occasion + 1) / 2

        # Weighted score
        occasion_similarity_weight = 0.7
        coherence_weight = 0.3
        final_match_score = (occasion_similarity_weight * similarity_to_occasion_normalized) + \
                            (coherence_weight * internal_coherence_score)

        scored_outfits.append({
            "outfit_model": outfit, # Keep the SQLAlchemy model
            "score": final_match_score,
            "debug_occasion_sim": similarity_to_occasion_normalized,
            "debug_coherence": internal_coherence_score
        })

    # Sort outfits by the final match score
    # Adding a secondary sort by coherence in case of tie in final_match_score, or just to favor more coherent ones slightly
    sorted_outfits = sorted(scored_outfits, key=lambda x: (x["score"], x["debug_coherence"]), reverse=True)

    # logger.debug(f"Sorted Outfits: {[ (s['outfit_model'].name, s['score'], s['debug_occasion_sim'],s['debug_coherence']) for s in sorted_outfits[:5]]}")


    return [s["outfit_model"] for s in sorted_outfits[:num_recommendations]]


async def recommend_outfits_for_occasion_service(
    db: Session,
    user: schemas.User, # User for whom recommendations are being made
    occasion: schemas.Occasion, # The occasion details
    num_recommendations: int = 3
) -> List[schemas.Outfit]: # Return a list of Outfit schemas

    occasion_text = f"{occasion.name} {occasion.notes if occasion.notes else ''}".strip()
    if not occasion_text: # Handle empty occasion details
        # Fallback: maybe return most coherent or recently created outfits?
        # For now, return empty if no text to match.
        return []

    # Use the new AI-driven helper
    db_outfits = find_ai_matched_outfits_for_occasion(
        db=db,
        user_id=user.id,
        occasion_text=occasion_text,
        num_recommendations=num_recommendations
    )

    recommendations = []
    for outfit_model in db_outfits:
        # The Outfit schema should be able to serialize from the SQLAlchemy model
        # Ensure that `schemas.Outfit.from_orm` or `model_validate` is configured correctly,
        # especially for nested items. The `joinedload` in the helper should make items available.
        try:
            recommendations.append(schemas.Outfit.model_validate(outfit_model))
        except Exception as e:
            logger.error(f"Error validating outfit model {outfit_model.id} for schema: {e}")
            # Optionally, skip this outfit or handle error
            continue

    return recommendations


async def get_wardrobe_recommendations_service(
    db: Session,
    user: schemas.User,
    num_recommendations: int = 5, # Max number of new outfit ideas
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> schemas.PersonalizedWardrobeSuggestions:

    weather_conditions: Optional[Dict[str, Any]] = None
    if latitude is not None and longitude is not None:
        # Assuming get_weather_data is async
        weather_conditions = await get_weather_data(latitude=latitude, longitude=longitude)
        # If get_weather_data is not async, call it directly:
        # weather_conditions = get_weather_data(latitude=latitude, longitude=longitude)

    user_items_query = db.query(models.WardrobeItem).filter(models.WardrobeItem.user_id == user.id)

    # Initial filtering based on very basic weather conditions if available
    # This is a simple approach. More advanced would involve scoring items.
    if weather_conditions:
        temp = weather_conditions.get("temperature_celsius")
        condition = weather_conditions.get("condition", "").lower()

        if temp is not None:
            if temp < 10: # Cold
                # Prioritize warmer clothes, allow layers
                # This example prioritizes by category or season, could also use tags
                user_items_query = user_items_query.filter(
                    or_(
                        models.WardrobeItem.category.in_(["Sweater", "Coat", "Jacket", "Outerwear", "Knitwear", "Hoodie", "Long-Sleeve"]),
                        models.WardrobeItem.season.in_(["Winter", "Autumn"]),
                        # Add tag-based filtering here if tags are well-defined for warmth
                    )
                )
            elif temp > 25: # Hot
                user_items_query = user_items_query.filter(
                    or_(
                        models.WardrobeItem.category.in_(["T-Shirt", "Shorts", "Tank Top", "Dress", "Skirt"]),
                        models.WardrobeItem.season.in_(["Summer", "Spring"]),
                    )
                )

        if "rain" in condition:
            # Further prioritize or filter for rain-proof items
            # This assumes 'Raincoat' category or a 'waterproof' tag exists
            # For simplicity, this example doesn't add a separate mandatory filter for rain anorak
            # but one could add an additional filter or boost scores of rain-appropriate items.
            pass # Placeholder for more specific rain logic, e.g. boosting raincoats

        if "snow" in condition:
            # Similar to cold, but could be more specific for snow gear
            user_items_query = user_items_query.filter(
                models.WardrobeItem.category.in_(["Winter Coat", "Insulated Jacket", "Boots", "Snow Pants"])
            )


    user_items = user_items_query.all()

    # Process items to ensure they have AI features (mocked if not present)
    processed_user_items: List[Dict[str, Any]] = []
    for item in user_items: # Corrected variable name from user_items_from_db
        # Use actual AI embedding if available, otherwise mock
        embedding = item.ai_embedding
        if embedding is None: # Check covers both missing attr and attr is None
            embedding = np.random.rand(512).tolist() # Example ViT-base embedding

        # Use actual AI dominant colors if available, otherwise mock
        ai_colors = item.ai_dominant_colors
        if ai_colors is None:
            ai_colors = random.sample(["#1A1A1A", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF", "#F0F0F0"], k=2)

        # Ensure category is present, mock if not
        category = getattr(item, "category", None)
        if category is None:
            category = random.choice(["Tops", "Bottoms", "Shoes", "Accessories", "Outerwear"])

        processed_user_items.append({
            "id": item.id,
            "name": item.name,
            "image_url": item.image_url, # Keep other useful fields
            "category": category,
            "embedding": embedding,
            "colors": ai_colors,
            # Add other fields like item.ai_style if it were available and needed
        })

    new_outfit_ideas: List[str] = []

    # --- "New Outfit Ideas" Generation ---
    # Filter items that have necessary features for matching
    matchable_items = [item for item in processed_user_items if item.get("embedding") and item.get("colors") and item.get("category")]

    # Group items by category for easier selection
    items_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for item in matchable_items:
        items_by_category.setdefault(item["category"], []).append(item)

    # Define typical outfit structures (e.g., top + bottom, top + bottom + shoes)
    # This is a simplified approach. More complex logic could handle more variations.
    outfit_structures = [
        ["Tops", "Bottoms"],
        ["Tops", "Bottoms", "Shoes"],
        ["Tops", "Bottoms", "Outerwear"],
    ]

    attempts_to_find_outfits = 15 # Try a number of times to generate diverse outfits
    generated_outfit_count = 0

    if len(matchable_items) >= 2: # Need at least 2 items to form an outfit
        for _ in range(attempts_to_find_outfits):
            if generated_outfit_count >= num_recommendations:
                break

            chosen_structure = random.choice(outfit_structures)
            current_outfit_items_features: List[Dict[str, Any]] = []
            current_outfit_item_names: List[str] = []

            possible_to_form = True
            for cat in chosen_structure:
                if items_by_category.get(cat):
                    chosen_item = random.choice(items_by_category[cat])
                    # Avoid choosing the same item twice if a category is listed multiple times (not in current structures)
                    if chosen_item["id"] not in [ci["id"] for ci in current_outfit_items_features]:
                        current_outfit_items_features.append(chosen_item)
                        current_outfit_item_names.append(chosen_item["name"])
                    else: # Could not find a unique item for this part of the structure
                        possible_to_form = False; break
                else: # Not enough items in a required category
                    possible_to_form = False; break

            if possible_to_form and len(current_outfit_items_features) >= 2:
                # Check if this specific combination of item names has been suggested already
                # Sort names to ensure order doesn't matter for uniqueness
                sorted_item_names = ", ".join(sorted(current_outfit_item_names))
                if any(sorted_item_names in idea for idea in new_outfit_ideas): # Basic check to avoid exact duplicates by name
                    continue

                score_details = outfit_matcher.calculate_compatibility_score(current_outfit_items_features)
                if score_details["score"] > 0.55: # Compatibility threshold
                    idea = (f"Try combining: {', '.join(current_outfit_item_names)} "
                            f"(Style: {score_details['style_cohesion_score']:.2f}, "
                            f"Color: {score_details['color_harmony_score']:.2f}, "
                            f"Overall: {score_details['score']:.2f})")
                    new_outfit_ideas.append(idea)
                    generated_outfit_count += 1

    if not new_outfit_ideas and matchable_items: # Fallback if no high-scoring outfits found
        new_outfit_ideas.append("Try experimenting with different combinations from your wardrobe! Use items from different categories like Tops, Bottoms, and Shoes.")


    # --- "Items to Acquire" Enhancement ---
    items_to_acquire_suggestions: List[str] = []
    user_categories_owned = {item["category"] for item in processed_user_items if item.get("category")}
    essential_categories = {"Tops", "Bottoms", "Shoes", "Outerwear"}
    missing_essentials = essential_categories - user_categories_owned

    for cat in missing_essentials:
        items_to_acquire_suggestions.append(f"Consider adding a versatile item to your '{cat}' collection (e.g., a neutral-colored {cat.lower()[:-1] if cat.endswith('s') else cat.lower()}).")

    # Basic trend integration placeholder (can be expanded if get_fashion_trends_service is ready)
    # try:
    #     trends_response = await get_fashion_trends_service(db, user) # Assuming it's async
    #     if trends_response and trends_response.trends:
    #         # Example: Suggest a trending item if user is missing its category
    #         for trend_item in trends_response.trends[:2]: # Check top 2 trends
    #             if trend_item.category in missing_essentials and len(items_to_acquire_suggestions) < 3:
    #                 items_to_acquire_suggestions.append(f"Fashion Highlight: '{trend_item.name}' ({trend_item.category}) is currently trending. This could be a great addition!")
    # except Exception as e:
    #     logger.error(f"Could not fetch fashion trends for recommendations: {e}")

    if not items_to_acquire_suggestions:
        items_to_acquire_suggestions.append("Your wardrobe essentials seem covered! Explore accessories to diversify your looks.")


    return schemas.PersonalizedWardrobeSuggestions(
        newOutfitIdeas=new_outfit_ideas[:num_recommendations],
        itemsToAcquire=items_to_acquire_suggestions[:3] # Limit acquisition suggestions
    )
