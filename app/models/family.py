from datetime import datetime
from sqlalchemy import String, Date, DateTime, ForeignKey, Boolean, Enum as SQLEnum, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

from app.models.base import Base


class RelationshipTypeEnum(str, enum.Enum):
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    GRANDPARENT = "grandparent"
    GRANDCHILD = "grandchild"
    AUNT_UNCLE = "aunt_uncle"
    NIECE_NEPHEW = "niece_nephew"
    COUSIN = "cousin"
    SPOUSE = "spouse"


class SeverityEnum(str, enum.Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class FamilyRelationship(Base):
    __tablename__ = "family_relationships"
    __table_args__ = (
        UniqueConstraint('patient_id', 'relative_patient_id', 'relationship_type', name='uq_family_relationship'),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False
    )
    relative_patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False
    )
    relationship_type: Mapped[RelationshipTypeEnum] = mapped_column(
        SQLEnum(RelationshipTypeEnum, native_enum=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    is_blood_relative: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship(
        "Patient", foreign_keys=[patient_id], back_populates="family_relationships"
    )
    relative_patient = relationship(
        "Patient", foreign_keys=[relative_patient_id], back_populates="relative_relationships"
    )

    def __repr__(self) -> str:
        return f"<FamilyRelationship(patient_id={self.patient_id}, relative={self.relative_patient_id}, type={self.relationship_type})>"


class FamilyDiseaseHistory(Base):
    __tablename__ = "family_disease_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False
    )
    disease_name: Mapped[str] = mapped_column(String(200), nullable=False)
    diagnosed_at: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    severity: Mapped[SeverityEnum | None] = mapped_column(
        SQLEnum(SeverityEnum, native_enum=False, values_callable=lambda x: [e.value for e in x]),
        nullable=True
    )
    notes: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="disease_history")

    def __repr__(self) -> str:
        return f"<FamilyDiseaseHistory(patient_id={self.patient_id}, disease={self.disease_name})>"

