"""
Find patients with the LONGEST disease progression timelines (any disease)
"""
import asyncio
from sqlalchemy import select, func
from app.db.session import AsyncSessionLocal
from app.models.disease import DiseaseProgression

async def find_longest_timelines():
    """Find patients with the most progression records (any disease)"""
    
    async with AsyncSessionLocal() as db:
        # Find ALL patients grouped by disease, sorted by record count
        query = select(
            DiseaseProgression.patient_id,
            DiseaseProgression.disease_name,
            func.count(DiseaseProgression.progression_id).label('record_count'),
            func.min(DiseaseProgression.assessed_date).label('first_date'),
            func.max(DiseaseProgression.assessed_date).label('last_date')
        ).group_by(
            DiseaseProgression.patient_id,
            DiseaseProgression.disease_name
        ).order_by(
            func.count(DiseaseProgression.progression_id).desc()  # Most records first
        ).limit(20)
        
        result = await db.execute(query)
        rows = result.all()
        
        if rows:
            print("\n" + "="*80)
            print("🔍 TOP 20 PATIENTS WITH LONGEST DISEASE TIMELINES (any disease)")
            print("="*80)
            
            best_patient = None
            best_disease = None
            best_count = 0
            
            for row in rows:
                patient_id = row[0]
                disease_name = row[1]
                record_count = row[2]
                first_date = row[3]
                last_date = row[4]
                
                # Calculate days span
                days_span = (last_date - first_date).days if last_date and first_date else 0
                
                print(f"\n📋 Patient ID: {patient_id}")
                print(f"   Disease: {disease_name}")
                print(f"   Records: {record_count}")
                print(f"   Timeline: {first_date.strftime('%Y-%m-%d')} → {last_date.strftime('%Y-%m-%d')} ({days_span} days)")
                
                if record_count > best_count:
                    best_count = record_count
                    best_patient = patient_id
                    best_disease = disease_name
                
                # Show the actual progression stages for this patient
                detail_query = select(
                    DiseaseProgression.progression_stage,
                    DiseaseProgression.assessed_date,
                    DiseaseProgression.confidence_score
                ).where(
                    DiseaseProgression.patient_id == patient_id,
                    DiseaseProgression.disease_name == disease_name
                ).order_by(DiseaseProgression.assessed_date.asc())
                
                detail_result = await db.execute(detail_query)
                stages = detail_result.all()
                
                print(f"   Stages: ", end="")
                stage_list = [f"{s[0]}" for s in stages[:8]]  # Show first 8
                print(" → ".join(stage_list))
                if len(stages) > 8:
                    print(f"   ... and {len(stages) - 8} more stages")
            
            print("\n" + "="*80)
            print(f"🎯 BEST PATIENT FOR TESTING (most records):")
            print(f"   Patient ID: {best_patient}")
            print(f"   Disease: {best_disease}")
            print(f"   Records: {best_count}")
            print("="*80)
            
            # Determine disease query param
            disease_param = best_disease.lower().replace(' ', '_') if best_disease else 'diabetes'
            if 'diabetes' in disease_param:
                disease_param = 'diabetes'
            elif 'anemia' in disease_param or 'iron' in disease_param:
                disease_param = 'anemia'
            elif 'kidney' in disease_param or 'ckd' in disease_param:
                disease_param = 'ckd'
            
            print(f"\n📌 Test with this curl command:")
            print(f"""
curl -X 'GET' \\
  'http://localhost:8080/api/v1/reports/patient/{best_patient}/progression-timeline?disease_name={disease_param}&months_back=36' \\
  -H 'accept: application/json'
""")
            
            return best_patient, best_disease, best_count
        else:
            print("\n❌ No disease progression data found in database at all!")
            
            # Show total count
            count_query = select(func.count(DiseaseProgression.progression_id))
            count_result = await db.execute(count_query)
            total = count_result.scalar()
            print(f"Total progression records in DB: {total}")
            
            return None, None, 0

if __name__ == "__main__":
    asyncio.run(find_longest_timelines())
