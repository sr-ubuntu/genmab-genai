"""Database initialisation and pre-seeded mock data.

This module provides functions to create the database schema and seed it with
realistic synthetic patient and clinical record data.

Run directly with: python -m clinical_summarizer.db.seed
"""
from __future__ import annotations
import datetime
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from clinical_summarizer.db.models import Base, Patient, ClinicalRecord, RecordType


def create_tables(engine: Engine) -> None:
    """Create all ORM-mapped tables if they do not already exist.

    Args:
        engine: A bound SQLAlchemy Engine instance.
    """
    Base.metadata.create_all(engine)


def seed_database(session: Session) -> None:
    """Insert pre-defined mock patients and clinical records.

    Inserts 4 patients with 2-3 records each (mixed lab_result and
    imaging_report types) using realistic synthetic clinical text.
    Idempotent — skips insertion if data already exists.

    Args:
        session: An open SQLAlchemy Session to use for inserts.
    """
    # Check if data already exists (idempotency)
    existing_count = session.query(Patient).count()
    if existing_count > 0:
        return

    # Patient 1: Alice Mercer
    p1 = Patient(
        name="Alice Mercer",
        date_of_birth=datetime.date(1968, 4, 12),
        gender="Female",
        mrn="MRN-001",
    )
    session.add(p1)
    session.flush()

    r1_1 = ClinicalRecord(
        patient_id=p1.id,
        record_type=RecordType.LAB_RESULT,
        title="Complete Blood Count",
        content=(
            "Patient: Alice Mercer, DOB: 1968-04-12\n"
            "Date: 2024-03-10\n"
            "Test: Complete Blood Count (CBC)\n\n"
            "Results:\n"
            "WBC: 7.2 K/uL (Normal: 4.5-11.0)\n"
            "RBC: 4.8 M/uL (Normal: 4.0-5.5)\n"
            "Hemoglobin: 14.2 g/dL (Normal: 12.0-16.0)\n"
            "Hematocrit: 42.5% (Normal: 36-46%)\n"
            "Platelets: 245 K/uL (Normal: 150-400)\n\n"
            "Interpretation: All values within normal range. No abnormalities detected."
        ),
        recorded_at=datetime.datetime(2024, 3, 10, 9, 30),
    )
    session.add(r1_1)

    r1_2 = ClinicalRecord(
        patient_id=p1.id,
        record_type=RecordType.IMAGING_REPORT,
        title="Chest X-Ray",
        content=(
            "Patient: Alice Mercer, DOB: 1968-04-12\n"
            "Date: 2024-03-15\n"
            "Modality: Chest X-Ray\n\n"
            "Findings:\n"
            "The lungs are clear with no evidence of pneumonia or atelectasis. "
            "Cardiac silhouette is normal in size and contour. Mediastinal outline is unremarkable. "
            "The trachea is midline. No acute osseous abnormalities noted.\n\n"
            "Impression:\n"
            "No acute thoracic pathology."
        ),
        recorded_at=datetime.datetime(2024, 3, 15, 14, 0),
    )
    session.add(r1_2)

    r1_3 = ClinicalRecord(
        patient_id=p1.id,
        record_type=RecordType.LAB_RESULT,
        title="Lipid Panel",
        content=(
            "Patient: Alice Mercer, DOB: 1968-04-12\n"
            "Date: 2024-03-12\n"
            "Test: Lipid Panel\n\n"
            "Results:\n"
            "Total Cholesterol: 195 mg/dL (Normal: <200)\n"
            "LDL Cholesterol: 110 mg/dL (Normal: <100)\n"
            "HDL Cholesterol: 52 mg/dL (Normal: >40 for females)\n"
            "Triglycerides: 120 mg/dL (Normal: <150)\n\n"
            "Interpretation: Slightly elevated LDL. Recommend dietary modifications."
        ),
        recorded_at=datetime.datetime(2024, 3, 12, 10, 15),
    )
    session.add(r1_3)

    # Patient 2: David Okafor
    p2 = Patient(
        name="David Okafor",
        date_of_birth=datetime.date(1975, 9, 30),
        gender="Male",
        mrn="MRN-002",
    )
    session.add(p2)
    session.flush()

    r2_1 = ClinicalRecord(
        patient_id=p2.id,
        record_type=RecordType.LAB_RESULT,
        title="Metabolic Panel",
        content=(
            "Patient: David Okafor, DOB: 1975-09-30\n"
            "Date: 2024-03-08\n"
            "Test: Comprehensive Metabolic Panel\n\n"
            "Results:\n"
            "Glucose: 102 mg/dL (Normal: 70-100) - ABNORMAL\n"
            "BUN: 18 mg/dL (Normal: 7-20)\n"
            "Creatinine: 1.0 mg/dL (Normal: 0.7-1.3)\n"
            "Sodium: 138 mEq/L (Normal: 136-145)\n"
            "Potassium: 4.2 mEq/L (Normal: 3.5-5.0)\n\n"
            "Interpretation: Elevated fasting glucose. Patient should follow up with "
            "endocrinology for diabetes screening."
        ),
        recorded_at=datetime.datetime(2024, 3, 8, 8, 0),
    )
    session.add(r2_1)

    r2_2 = ClinicalRecord(
        patient_id=p2.id,
        record_type=RecordType.IMAGING_REPORT,
        title="Abdominal CT Scan",
        content=(
            "Patient: David Okafor, DOB: 1975-09-30\n"
            "Date: 2024-03-18\n"
            "Modality: CT Abdomen and Pelvis with Contrast\n\n"
            "Findings:\n"
            "Liver: Normal size and attenuation. No focal lesions or cirrhotic changes. "
            "Pancreas: Normal size and echotexture. No ductal dilatation. "
            "Spleen: Normal size. Kidneys: Bilateral, symmetric, normal enhancement. "
            "No hydronephrosis. No evidence of aortic aneurysm or dissection.\n\n"
            "Impression:\n"
            "No acute abdominal or pelvic pathology. Liver appears unremarkable."
        ),
        recorded_at=datetime.datetime(2024, 3, 18, 11, 30),
    )
    session.add(r2_2)

    # Patient 3: Susan Park
    p3 = Patient(
        name="Susan Park",
        date_of_birth=datetime.date(1982, 1, 17),
        gender="Female",
        mrn="MRN-003",
    )
    session.add(p3)
    session.flush()

    r3_1 = ClinicalRecord(
        patient_id=p3.id,
        record_type=RecordType.LAB_RESULT,
        title="Thyroid Function Tests",
        content=(
            "Patient: Susan Park, DOB: 1982-01-17\n"
            "Date: 2024-03-05\n"
            "Test: Thyroid Panel\n\n"
            "Results:\n"
            "TSH: 0.8 mIU/L (Normal: 0.5-5.0)\n"
            "Free T4: 8.5 ng/dL (Normal: 6.5-10.5)\n"
            "Free T3: 3.1 pg/mL (Normal: 2.5-3.9)\n\n"
            "Interpretation: Thyroid function is normal. No evidence of hypothyroidism or hyperthyroidism."
        ),
        recorded_at=datetime.datetime(2024, 3, 5, 10, 0),
    )
    session.add(r3_1)

    r3_2 = ClinicalRecord(
        patient_id=p3.id,
        record_type=RecordType.IMAGING_REPORT,
        title="Mammography",
        content=(
            "Patient: Susan Park, DOB: 1982-01-17\n"
            "Date: 2024-03-20\n"
            "Modality: Bilateral Mammography\n\n"
            "Findings:\n"
            "Left breast: Fibroglandular tissue. No suspicious masses, microcalcifications, "
            "or architectural distortions noted. Right breast: Similar to left. "
            "No findings of concern. BI-RADS Category 1.\n\n"
            "Impression:\n"
            "Negative mammogram. Routine screening recommended."
        ),
        recorded_at=datetime.datetime(2024, 3, 20, 15, 45),
    )
    session.add(r3_2)

    r3_3 = ClinicalRecord(
        patient_id=p3.id,
        record_type=RecordType.LAB_RESULT,
        title="Urinalysis",
        content=(
            "Patient: Susan Park, DOB: 1982-01-17\n"
            "Date: 2024-03-06\n"
            "Test: Urinalysis\n\n"
            "Results:\n"
            "Color: Pale Yellow\n"
            "Clarity: Clear\n"
            "pH: 6.5\n"
            "Specific Gravity: 1.015\n"
            "Glucose: Negative\n"
            "Protein: Negative\n"
            "RBC: Negative\n"
            "WBC: Negative\n"
            "Nitrites: Negative\n\n"
            "Interpretation: Completely normal urinalysis. No UTI or proteinuria detected."
        ),
        recorded_at=datetime.datetime(2024, 3, 6, 9, 30),
    )
    session.add(r3_3)

    # Patient 4: James Rivera
    p4 = Patient(
        name="James Rivera",
        date_of_birth=datetime.date(1955, 7, 22),
        gender="Male",
        mrn="MRN-004",
    )
    session.add(p4)
    session.flush()

    r4_1 = ClinicalRecord(
        patient_id=p4.id,
        record_type=RecordType.LAB_RESULT,
        title="Prostate Specific Antigen (PSA)",
        content=(
            "Patient: James Rivera, DOB: 1955-07-22\n"
            "Date: 2024-03-11\n"
            "Test: PSA\n\n"
            "Result: 2.8 ng/mL (Normal for age 68: <4.0)\n\n"
            "Interpretation: PSA is within normal limits. No evidence of prostate cancer. "
            "Patient should continue routine screening as per guidelines."
        ),
        recorded_at=datetime.datetime(2024, 3, 11, 13, 0),
    )
    session.add(r4_1)

    r4_2 = ClinicalRecord(
        patient_id=p4.id,
        record_type=RecordType.IMAGING_REPORT,
        title="Colonoscopy Report",
        content=(
            "Patient: James Rivera, DOB: 1955-07-22\n"
            "Date: 2024-03-25\n"
            "Procedure: Colonoscopy\n\n"
            "Findings:\n"
            "Cecum: Visualized. No lesions. Ascending colon: Unremarkable mucosa. "
            "Transverse colon: No polyps or masses. Descending colon: Mucosa appears normal. "
            "Sigmoid colon: One small hyperplastic polyp 5mm, removed via cold snare. "
            "Rectum: Normal.\n\n"
            "Impression:\n"
            "Successful colonoscopy with removal of one small hyperplastic polyp. "
            "No other significant findings. Next colonoscopy in 10 years."
        ),
        recorded_at=datetime.datetime(2024, 3, 25, 10, 0),
    )
    session.add(r4_2)

    session.commit()


def init_db(db_url: str = "sqlite:///clinical.db") -> None:
    """Initialize the database: create tables and seed with mock data.

    Args:
        db_url: SQLAlchemy connection string.
    """
    from clinical_summarizer.db import get_engine, get_session_factory

    engine = get_engine(db_url)
    create_tables(engine)
    session_factory = get_session_factory(engine)
    session = session_factory()
    try:
        seed_database(session)
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully with seed data.")
