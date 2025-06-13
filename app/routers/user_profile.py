from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from .. import tables as schemas  # schemas are in tables.py
from .. import model as models    # SQLAlchemy models are in model.py
from ..security import get_current_user
from ..db.database import get_db
from ..services.ai_style_insights_service import (
    get_user_style_profile,
    get_wardrobe_analysis_details,
    generate_personalized_general_insights,
    generate_ai_style_outfit_recommendations
)
# FullAIStyleInsightsResponse is already available via schemas.FullAIStyleInsightsResponse

router = APIRouter(
    prefix="/profile",
    tags=["User Profile"],
    responses={404: {"description": "Not found"}},
)

@router.get("/me", response_model=schemas.UserProfile)
async def read_user_profile(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == current_user.id).first()
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found for current user")
    return profile

@router.post("/me", response_model=schemas.UserProfile, status_code=status.HTTP_201_CREATED)
async def create_user_profile(
    profile_data: schemas.UserProfileCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    existing_profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == current_user.id).first()
    if existing_profile:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile already exists for this user. Use PUT to update.")

    new_profile = models.UserProfile(
        **profile_data.model_dump(),
        user_id=current_user.id,
        updated_at=datetime.utcnow()
    )
    db.add(new_profile)
    db.commit()
    db.refresh(new_profile)
    return new_profile

@router.put("/me", response_model=schemas.UserProfile)
async def update_user_profile(
    profile_update_data: schemas.UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    db_profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == current_user.id).first()

    if db_profile is None:
        # If profile doesn't exist, create it (idempotent PUT)
        new_profile = models.UserProfile(
            **profile_update_data.model_dump(exclude_unset=True), # include all fields from schema for creation
            user_id=current_user.id,
            updated_at=datetime.utcnow()
        )
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        return new_profile
    else:
        # Profile exists, update it
        update_data = profile_update_data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_profile, key, value)
        db_profile.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_profile)
        return db_profile

@router.get("/me/style-insights", response_model=schemas.FullAIStyleInsightsResponse)
async def get_full_style_insights(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Retrieves comprehensive AI-driven style insights for the current user.
    This includes their style profile, wardrobe analysis, personalized tips,
    and outfit recommendations.
    """
    user_style_profile_response = get_user_style_profile(db=db, user=current_user)
    wardrobe_analysis_response = get_wardrobe_analysis_details(db=db, user=current_user)

    personalized_insights_list = generate_personalized_general_insights(
        user_style_profile=user_style_profile_response,
        wardrobe_analysis=wardrobe_analysis_response
    )

    suggested_outfits_list = generate_ai_style_outfit_recommendations(
        db=db,
        user=current_user,
        user_style_profile=user_style_profile_response,
        wardrobe_analysis=wardrobe_analysis_response
    )

    return schemas.FullAIStyleInsightsResponse(
        user_profile=user_style_profile_response,
        wardrobe_analysis=wardrobe_analysis_response,
        personalized_insights=personalized_insights_list,
        suggested_outfits=suggested_outfits_list
    )
