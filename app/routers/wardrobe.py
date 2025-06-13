from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from typing import List, Optional
from datetime import datetime
import shutil
import uuid
import os
import logging # Added for logging
from PIL import Image
from sqlalchemy.orm import Session

from .. import tables as schemas
from .. import model as models # Import models and schemas
from ..services import ai_embedding, ai_services # Import AI services
from ..security import get_current_user # get_current_user returns schemas.User
from ..db.database import get_db

router = APIRouter(
    prefix="/wardrobe",
    tags=["Wardrobe"],
    responses={404: {"description": "Not found"}},
)

# Remove fake_wardrobe_db and next_item_id

# Define the static directory for images
STATIC_DIR = "static"
WARDROBE_IMAGES_DIR = os.path.join(STATIC_DIR, "wardrobe_images")
# Create the directory if it doesn't exist. This should ideally be done at app startup.
os.makedirs(WARDROBE_IMAGES_DIR, exist_ok=True)

logger = logging.getLogger(__name__) # Added logger

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp"]

@router.post("/items/", response_model=schemas.WardrobeItem, status_code=status.HTTP_201_CREATED)
async def create_wardrobe_item(
    item: schemas.WardrobeItemCreate = Depends(), # Use Depends for form data when file is also expected
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    item_data = item.model_dump() # Get data from the Pydantic model
    item_data['ai_embedding'] = None
    item_data['ai_dominant_colors'] = None

    # Handle image upload
    if image and image.filename:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image type. Allowed types: {ALLOWED_CONTENT_TYPES}")

        img_bytes = await image.read()
        await image.seek(0) # Reset cursor for saving
        if len(img_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Image too large. Max size: {MAX_FILE_SIZE_BYTES // (1024*1024)}MB")

        # Generate a unique filename
        unique_id = uuid.uuid4()
        extension = os.path.splitext(image.filename)[1]
        filename = f"{unique_id}{extension}"
        file_path = os.path.join(WARDROBE_IMAGES_DIR, filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            item_data['image_url'] = f"/{file_path}"  # Store relative path

            # AI processing
            pil_image = Image.open(file_path)
            try:
                embedding = ai_embedding.get_image_embedding(pil_image)
                item_data['ai_embedding'] = embedding
            except Exception as e:
                logger.error(f"Error generating image embedding: {e}")
                # Store None or handle as per specific requirement for embedding errors

            try:
                colors = ai_services.extract_colors(pil_image)
                item_data['ai_dominant_colors'] = colors
            except Exception as e:
                logger.error(f"Error extracting dominant colors: {e}")

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            item_data['image_url'] = item.image_url # Fallback to image_url from body if any
        finally:
            image.file.close()
    elif item.image_url: # If no new image uploaded, but image_url is in the payload
        item_data['image_url'] = item.image_url
    else:
        item_data['image_url'] = None


    db_item = models.WardrobeItem(
        **item_data, # image_url, ai_embedding, ai_dominant_colors are now part of item_data
        user_id=current_user.id,
        date_added=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        times_worn=0, # Default value
        favorite=item_data.get('favorite', False) # Ensure favorite has a default if not in item_data
    )
    # The WardrobeItem model's @tags.setter handles JSON conversion.
    # item_data from model_dump() on WardrobeItemCreate already includes tags if provided.
    # So, if item_data['tags'] is not None, it will be set correctly by **item_data.

    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@router.get("/items/", response_model=List[schemas.WardrobeItem])
async def read_wardrobe_items(
    category: Optional[str] = None,
    season: Optional[str] = None,
    favorite: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    query = db.query(models.WardrobeItem).filter(models.WardrobeItem.user_id == current_user.id)

    if category:
        query = query.filter(models.WardrobeItem.category.ilike(f"%{category}%"))
    if season:
        query = query.filter(models.WardrobeItem.season.ilike(f"%{season}%"))
    if favorite is not None:
        query = query.filter(models.WardrobeItem.favorite == favorite)

    items = query.offset(skip).limit(limit).all()
    return items

@router.get("/items/{item_id}", response_model=schemas.WardrobeItem)
async def read_wardrobe_item(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    db_item = db.query(models.WardrobeItem).filter(models.WardrobeItem.id == item_id, models.WardrobeItem.user_id == current_user.id).first()
    if db_item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    return db_item

@router.put("/items/{item_id}", response_model=schemas.WardrobeItem)
async def update_wardrobe_item(
    item_id: int,
    item_update: schemas.WardrobeItemUpdate = Depends(), # Use Depends for form data
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    db_item = db.query(models.WardrobeItem).filter(models.WardrobeItem.id == item_id, models.WardrobeItem.user_id == current_user.id).first()

    if db_item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")

    update_data = item_update.model_dump(exclude_unset=True)

    if image and image.filename: # A new image is being uploaded
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image type. Allowed types: {ALLOWED_CONTENT_TYPES}")

        img_bytes = await image.read()
        await image.seek(0) # Reset cursor for saving
        if len(img_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Image too large. Max size: {MAX_FILE_SIZE_BYTES // (1024*1024)}MB")

        # Optionally, delete the old image file if it exists
        if db_item.image_url:
            old_image_path_on_disk = db_item.image_url.lstrip("/")
            if os.path.exists(old_image_path_on_disk):
                try:
                    os.remove(old_image_path_on_disk)
                except Exception as e:
                    logger.error(f"Error deleting old image {old_image_path_on_disk}: {e}")

        # Save the new image
        unique_id = uuid.uuid4()
        extension = os.path.splitext(image.filename)[1]
        new_filename = f"{unique_id}{extension}"
        new_file_path_on_disk = os.path.join(WARDROBE_IMAGES_DIR, new_filename)

        try:
            with open(new_file_path_on_disk, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            update_data['image_url'] = f"/{new_file_path_on_disk}" # Update path for DB

            # AI processing for the new image
            pil_image = Image.open(new_file_path_on_disk)
            try:
                embedding = ai_embedding.get_image_embedding(pil_image)
                update_data['ai_embedding'] = embedding
            except Exception as e:
                logger.error(f"Error generating image embedding for update: {e}")
                update_data['ai_embedding'] = None

            try:
                colors = ai_services.extract_colors(pil_image)
                update_data['ai_dominant_colors'] = colors
            except Exception as e:
                logger.error(f"Error extracting dominant colors for update: {e}")
                update_data['ai_dominant_colors'] = None
        except Exception as e:
            logger.error(f"Error saving new image: {e}")
            # If saving new image fails, we might want to revert image_url or handle error
            # For now, if it fails here, image_url in update_data might not be set or be old
        finally:
            image.file.close()

    elif 'image_url' in update_data and update_data['image_url'] is None:
        # This case handles when 'image_url' is explicitly set to null in the request,
        # meaning the user wants to remove the existing image without uploading a new one.
        if db_item.image_url:
            old_image_path_on_disk = db_item.image_url.lstrip("/")
            if os.path.exists(old_image_path_on_disk):
                try:
                    os.remove(old_image_path_on_disk)
                except Exception as e:
                    logger.error(f"Error deleting image {old_image_path_on_disk}: {e}")
        # Ensure AI fields are also cleared if the image is removed
        update_data['ai_embedding'] = None
        update_data['ai_dominant_colors'] = None
    # If no new image is uploaded and image_url is not being set to None,
    # then existing image_url, ai_embedding, and ai_dominant_colors on db_item remain unchanged
    # unless explicitly part of item_update (which they are not for these fields).

    for key, value in update_data.items():
        setattr(db_item, key, value)

    db_item.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_item)
    return db_item

@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_wardrobe_item(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    db_item = db.query(models.WardrobeItem).filter(models.WardrobeItem.id == item_id, models.WardrobeItem.user_id == current_user.id).first()

    if db_item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")

    # Delete the image file if it exists
    if db_item.image_url:
        image_path_on_disk = db_item.image_url.lstrip("/")  # Remove leading '/'
        if os.path.exists(image_path_on_disk):
            try:
                os.remove(image_path_on_disk)
            except FileNotFoundError:
                logger.warning(f"File not found: {image_path_on_disk}")
            except Exception as e:
                logger.error(f"Error deleting file {image_path_on_disk}: {e}")
        else:
            logger.warning(f"Image path not found, but listed in DB: {image_path_on_disk}")


    db.delete(db_item)
    db.commit()
    return
