from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Table, Text, JSON
from sqlalchemy.orm import relationship, sessionmaker
#from sqlalchemy.ext.declarative import declarative_base # Base is imported, declarative_base not directly used
from datetime import datetime
#import json # SQLAlchemy's JSON type handles serialization
import json
from .db.database import Base # Import Base from the new database.py

# Association table for Outfit and WardrobeItem (many-to-many)
outfit_item_association = Table('outfit_item_association', Base.metadata,
    Column('outfit_id', Integer, ForeignKey('outfits.id'), primary_key=True),
    Column('wardrobe_item_id', Integer, ForeignKey('wardrobe_items.id'), primary_key=True)
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    wardrobe_items = relationship("WardrobeItem", back_populates="owner")
    outfits = relationship("Outfit", back_populates="owner")
    weekly_plans = relationship("WeeklyPlan", back_populates="owner")
    feedbacks = relationship("Feedback", back_populates="commenter")  # or 'user'
    occasions = relationship("Occasion", back_populates="owner")
    style_history_entries = relationship("StyleHistory", back_populates="user")
    profile = relationship("UserProfile", uselist=False, back_populates="owner", cascade="all, delete-orphan")


class WardrobeItem(Base):
    __tablename__ = "wardrobe_items"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False, index=True)
    brand = Column(String(255), nullable=True)
    category = Column(String(255), nullable=False, index=True)
    size = Column(String(50), nullable=True)
    price = Column(Float, nullable=True)
    material = Column(String(255), nullable=True)
    season = Column(String(50), nullable=True, index=True) # e.g., Summer, Winter, All Seasons
    image_url = Column(String(2048), nullable=True)
    ai_embedding = Column(JSON, nullable=True)
    ai_dominant_colors = Column(JSON, nullable=True)
    _tags = Column("tags", Text, nullable=True) # Store as JSON string
    favorite = Column(Boolean, default=False)
    times_worn = Column(Integer, default=0)
    date_added = Column(DateTime, default=datetime.utcnow)
    last_worn = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="wardrobe_items")
    outfits_associated = relationship("Outfit", secondary=outfit_item_association, back_populates="items")
    style_history_entries = relationship("StyleHistory", back_populates="item_worn")


    @property
    def tags(self):
        return json.loads(self._tags) if self._tags else []

    @tags.setter
    def tags(self, value):
        self._tags = json.dumps(value)


class Outfit(Base):
    __tablename__ = "outfits"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    _tags = Column("tags", Text, nullable=True) # Store as JSON string
    image_url = Column(String(2048), nullable=True) # For a composed image of the outfit

    owner = relationship("User", back_populates="outfits")
    items = relationship("WardrobeItem", secondary=outfit_item_association, back_populates="outfits_associated")
    style_history_entries = relationship("StyleHistory", back_populates="outfit_worn")
    occasions_linked = relationship("Occasion", back_populates="outfit_assigned") # An outfit can be assigned to multiple occasions
    weekly_plan_days = relationship("WeeklyPlanDayOutfit", back_populates="outfit")
    feedbacks = relationship("Feedback", back_populates="outfit")

    @property
    def tags(self):
        return json.loads(self._tags) if self._tags else []

    @tags.setter
    def tags(self, value):
        self._tags = json.dumps(value)

    @property
    def item_ids(self):
        return [item.id for item in self.items]


class WeeklyPlan(Base):
    __tablename__ = "weekly_plans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False) # e.g., "Work Week Outfits", "Vacation Plan"
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="weekly_plans")
    daily_outfits = relationship("WeeklyPlanDayOutfit", back_populates="weekly_plan", cascade="all, delete-orphan")


class WeeklyPlanDayOutfit(Base):
    __tablename__ = "weekly_plan_day_outfits"
    id = Column(Integer, primary_key=True, index=True)
    weekly_plan_id = Column(Integer, ForeignKey("weekly_plans.id"), nullable=False)
    day_of_week = Column(String(10), nullable=False) # e.g., "monday", "tuesday"
    outfit_id = Column(Integer, ForeignKey("outfits.id"), nullable=True)

    weekly_plan = relationship("WeeklyPlan", back_populates="daily_outfits")
    outfit = relationship("Outfit", back_populates="weekly_plan_days")


class Occasion(Base):
    __tablename__ = "occasions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False, index=True) # E.g., "Wedding Guest", "Beach Party"
    date = Column(DateTime, nullable=True)
    outfit_id = Column(Integer, ForeignKey("outfits.id"), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="occasions")
    outfit_assigned = relationship("Outfit", back_populates="occasions_linked")


class StyleHistory(Base):
    __tablename__ = "style_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    item_id = Column(Integer, ForeignKey("wardrobe_items.id"), nullable=True)
    outfit_id = Column(Integer, ForeignKey("outfits.id"), nullable=True)
    date_worn = Column(DateTime, nullable=False, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="style_history_entries")
    item_worn = relationship("WardrobeItem", back_populates="style_history_entries")
    outfit_worn = relationship("Outfit", back_populates="style_history_entries")


class UserProfile(Base):
    __tablename__ = "user_profiles"
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True, index=True)
    preferred_styles = Column(JSON, nullable=True)  # Stores List[str]
    preferred_colors = Column(JSON, nullable=True)  # Stores List[str]
    avoided_colors = Column(JSON, nullable=True)  # Stores List[str]
    sizes = Column(JSON, nullable=True)  # Stores Dict[str, str]
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="profile")


class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    outfit_id = Column(Integer, ForeignKey("outfits.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # User who gave the feedback
    feedback_text = Column(Text, nullable=True)
    rating = Column(Integer, nullable=True) # e.g., 1-5 stars
    created_at = Column(DateTime, default=datetime.utcnow)

    outfit = relationship("Outfit", back_populates="feedbacks")
    commenter = relationship("User", back_populates="feedbacks")  # or rename to 'user'



# Example usage for creating tables (typically in main.py or a setup script)
# if __name__ == "__main__":
#     from .db.database import engine
#     Base.metadata.create_all(bind=engine)
