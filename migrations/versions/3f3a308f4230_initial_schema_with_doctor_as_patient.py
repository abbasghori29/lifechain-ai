"""initial_schema_with_doctor_as_patient

Revision ID: 3f3a308f4230
Revises: 
Create Date: 2025-11-22 21:01:23.569726

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '3f3a308f4230'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop existing enum types if they exist (from previous failed attempts)
    op.execute('DROP TYPE IF EXISTS genderenum CASCADE')
    op.execute('DROP TYPE IF EXISTS visittypeenum CASCADE')
    op.execute('DROP TYPE IF EXISTS diagnosisstatusenum CASCADE')
    op.execute('DROP TYPE IF EXISTS reportstatusenum CASCADE')
    op.execute('DROP TYPE IF EXISTS relationshiptypeenum CASCADE')
    op.execute('DROP TYPE IF EXISTS severityenum CASCADE')
    
    # Create enum types using raw SQL to avoid SQLAlchemy's automatic creation
    op.execute("CREATE TYPE genderenum AS ENUM ('male', 'female', 'other')")
    op.execute("CREATE TYPE visittypeenum AS ENUM ('consultation', 'follow_up', 'routine_checkup', 'lab_review', 'emergency')")
    op.execute("CREATE TYPE diagnosisstatusenum AS ENUM ('suspected', 'confirmed')")
    op.execute("CREATE TYPE reportstatusenum AS ENUM ('pending', 'completed')")
    op.execute("CREATE TYPE relationshiptypeenum AS ENUM ('parent', 'child', 'sibling', 'grandparent', 'grandchild', 'aunt_uncle', 'niece_nephew', 'cousin', 'spouse')")
    op.execute("CREATE TYPE severityenum AS ENUM ('mild', 'moderate', 'severe')")
    
    # Create enum objects for use in table definitions (create_type=False since we already created them)
    gender_enum = postgresql.ENUM('male', 'female', 'other', name='genderenum', create_type=False)
    visit_type_enum = postgresql.ENUM('consultation', 'follow_up', 'routine_checkup', 'lab_review', 'emergency', name='visittypeenum', create_type=False)
    diagnosis_status_enum = postgresql.ENUM('suspected', 'confirmed', name='diagnosisstatusenum', create_type=False)
    report_status_enum = postgresql.ENUM('pending', 'completed', name='reportstatusenum', create_type=False)
    relationship_type_enum = postgresql.ENUM('parent', 'child', 'sibling', 'grandparent', 'grandchild', 'aunt_uncle', 'niece_nephew', 'cousin', 'spouse', name='relationshiptypeenum', create_type=False)
    severity_enum = postgresql.ENUM('mild', 'moderate', 'severe', name='severityenum', create_type=False)
    
    # Create labs table
    op.create_table('labs',
        sa.Column('lab_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('lab_name', sa.String(length=200), nullable=False),
        sa.Column('lab_location', sa.String(length=500), nullable=True),
        sa.Column('accreditation_number', sa.String(length=100), nullable=True),
        sa.Column('phone', sa.String(length=20), nullable=True),
        sa.Column('email', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('lab_id')
    )
    
    # Create patients table
    op.create_table('patients',
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('cnic', sa.String(length=15), nullable=False),
        sa.Column('first_name', sa.String(length=100), nullable=False),
        sa.Column('last_name', sa.String(length=100), nullable=False),
        sa.Column('date_of_birth', sa.Date(), nullable=False),
        sa.Column('gender', gender_enum, nullable=False),
        sa.Column('blood_group', sa.String(length=5), nullable=True),
        sa.Column('phone', sa.String(length=20), nullable=True),
        sa.Column('email', sa.String(length=100), nullable=True),
        sa.Column('address', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_doctor', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('specialization', sa.String(length=100), nullable=True),
        sa.Column('license_number', sa.String(length=50), nullable=True),
        sa.Column('hospital_affiliation', sa.String(length=200), nullable=True),
        sa.PrimaryKeyConstraint('patient_id'),
        sa.UniqueConstraint('cnic'),
        sa.UniqueConstraint('license_number')
    )
    op.create_index('ix_patients_cnic', 'patients', ['cnic'], unique=False)
    op.create_index('ix_patients_license_number', 'patients', ['license_number'], unique=False)
    
    # Create family_relationships table
    op.create_table('family_relationships',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('relative_patient_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('relationship_type', relationship_type_enum, nullable=False),
        sa.Column('is_blood_relative', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['relative_patient_id'], ['patients.patient_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('patient_id', 'relative_patient_id', 'relationship_type', name='uq_family_relationship')
    )
    
    # Create family_disease_history table
    op.create_table('family_disease_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('disease_name', sa.String(length=200), nullable=False),
        sa.Column('diagnosed_at', sa.Date(), nullable=True),
        sa.Column('severity', severity_enum, nullable=True),
        sa.Column('notes', sa.String(length=1000), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create doctor_visits table
    op.create_table('doctor_visits',
        sa.Column('visit_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('doctor_patient_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('visit_date', sa.DateTime(), nullable=False),
        sa.Column('chief_complaint', sa.Text(), nullable=True),
        sa.Column('visit_type', visit_type_enum, nullable=False),
        sa.Column('doctor_notes', sa.Text(), nullable=True),
        sa.Column('vital_signs', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['doctor_patient_id'], ['patients.patient_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('visit_id')
    )
    
    # Create symptoms table
    op.create_table('symptoms',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('visit_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('symptom_name', sa.String(length=200), nullable=False),
        sa.Column('severity', sa.Integer(), nullable=True),
        sa.Column('duration_days', sa.Integer(), nullable=True),
        sa.Column('notes', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['visit_id'], ['doctor_visits.visit_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create diagnoses table
    op.create_table('diagnoses',
        sa.Column('diagnosis_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('visit_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('disease_name', sa.String(length=200), nullable=False),
        sa.Column('diagnosis_date', sa.DateTime(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('ml_model_used', sa.String(length=100), nullable=True),
        sa.Column('status', diagnosis_status_enum, nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['visit_id'], ['doctor_visits.visit_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('diagnosis_id')
    )
    
    # Create prescriptions table
    op.create_table('prescriptions',
        sa.Column('prescription_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('visit_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('medication_name', sa.String(length=200), nullable=False),
        sa.Column('dosage', sa.String(length=100), nullable=False),
        sa.Column('frequency', sa.String(length=100), nullable=False),
        sa.Column('duration_days', sa.Integer(), nullable=True),
        sa.Column('instructions', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['visit_id'], ['doctor_visits.visit_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('prescription_id')
    )
    
    # Create lab_reports table
    op.create_table('lab_reports',
        sa.Column('report_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('lab_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('visit_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('report_date', sa.DateTime(), nullable=False),
        sa.Column('report_type', sa.String(length=100), nullable=False),
        sa.Column('status', report_status_enum, nullable=False),
        sa.Column('pdf_url', sa.String(length=500), nullable=True),
        sa.Column('test_name', sa.String(length=200), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['lab_id'], ['labs.lab_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['visit_id'], ['doctor_visits.visit_id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('report_id')
    )
    
    # Create lab_test_results table
    op.create_table('lab_test_results',
        sa.Column('result_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('report_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('test_name', sa.String(length=200), nullable=False),
        sa.Column('test_value', sa.Float(), nullable=False),
        sa.Column('unit', sa.String(length=50), nullable=False),
        sa.Column('reference_range_min', sa.Float(), nullable=True),
        sa.Column('reference_range_max', sa.Float(), nullable=True),
        sa.Column('is_abnormal', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['report_id'], ['lab_reports.report_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('result_id')
    )
    
    # Create disease_progressions table
    op.create_table('disease_progressions',
        sa.Column('progression_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('disease_name', sa.String(length=200), nullable=False),
        sa.Column('progression_stage', sa.String(length=100), nullable=False),
        sa.Column('assessed_date', sa.DateTime(), nullable=False),
        sa.Column('ml_model_used', sa.String(length=100), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('notes', sa.String(length=1000), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('progression_id')
    )
    
    # Create ml_predictions table
    op.create_table('ml_predictions',
        sa.Column('prediction_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('input_features', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('prediction_result', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('prediction_date', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('prediction_id')
    )


def downgrade() -> None:
    # Drop tables in reverse order (with IF EXISTS to handle missing tables)
    op.execute('DROP TABLE IF EXISTS ml_predictions CASCADE')
    op.execute('DROP TABLE IF EXISTS disease_progressions CASCADE')
    op.execute('DROP TABLE IF EXISTS lab_test_results CASCADE')
    op.execute('DROP TABLE IF EXISTS lab_reports CASCADE')
    op.execute('DROP TABLE IF EXISTS prescriptions CASCADE')
    op.execute('DROP TABLE IF EXISTS diagnoses CASCADE')
    op.execute('DROP TABLE IF EXISTS symptoms CASCADE')
    op.execute('DROP TABLE IF EXISTS doctor_visits CASCADE')
    op.execute('DROP TABLE IF EXISTS family_disease_history CASCADE')
    op.execute('DROP TABLE IF EXISTS family_relationships CASCADE')
    op.execute('DROP TABLE IF EXISTS patients CASCADE')
    op.execute('DROP TABLE IF EXISTS labs CASCADE')
    
    # Drop enum types
    op.execute('DROP TYPE IF EXISTS severityenum CASCADE')
    op.execute('DROP TYPE IF EXISTS relationshiptypeenum CASCADE')
    op.execute('DROP TYPE IF EXISTS reportstatusenum CASCADE')
    op.execute('DROP TYPE IF EXISTS diagnosisstatusenum CASCADE')
    op.execute('DROP TYPE IF EXISTS visittypeenum CASCADE')
    op.execute('DROP TYPE IF EXISTS genderenum CASCADE')
