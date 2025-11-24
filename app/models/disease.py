from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from app.models.base import Base


class DiseaseProgression(Base):
    __tablename__ = "disease_progressions"

    progression_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False
    )
    disease_name: Mapped[str] = mapped_column(String(200), nullable=False)
    progression_stage: Mapped[str] = mapped_column(String(100), nullable=False)
    assessed_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ml_model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="disease_progressions")

    def __repr__(self) -> str:
        return f"<DiseaseProgression(disease={self.disease_name}, stage={self.progression_stage}, date={self.assessed_date})>"


class MLPrediction(Base):
    __tablename__ = "ml_predictions"

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    input_features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    prediction_result: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="ml_predictions")

    def __repr__(self) -> str:
        return f"<MLPrediction(model={self.model_name}, patient_id={self.patient_id}, date={self.prediction_date})>"

