from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

def pretty_print_situation(situation):
    """Pretty print the CBT situation analysis"""
    print("\n" + "="*60)
    print("🧠 CBT SITUATION ANALYSIS")
    print("="*60)
    
    # Timestamp
    print(f"📅 When: {situation.when.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Situation
    print(f"\n📋 SITUATION:")
    if situation.situation:
        print(f"   {situation.situation}")
    else:
        print("   (No situation details captured)")
    
    # Emotions
    print(f"\n💭 EMOTIONS:")
    if situation.emotions:
        for emotion in situation.emotions:
            print(f"   • {emotion}")
    else:
        print("   (No emotions identified)")
    
    # Automatic Thoughts
    print(f"\n🔄 AUTOMATIC THOUGHTS:")
    if situation.automatic_thoughts:
        for i, thought in enumerate(situation.automatic_thoughts, 1):
            print(f"   {i}. {thought}")
    else:
        print("   (No automatic thoughts captured)")
    
    # Meaning of Thoughts
    print(f"\n🎯 MEANING OF THOUGHTS:")
    if situation.meaning_of_at:
        print(f"   {situation.meaning_of_at}")
    else:
        print("   (Meaning not explored yet)")
    
    # Behaviors
    print(f"\n⚡ BEHAVIORS/RESPONSES:")
    if situation.behavior:
        for behavior in situation.behavior:
            print(f"   • {behavior}")
    else:
        print("   (No behaviors identified)")
    
    print("\n" + "="*60)
    
    # Summary stats
    completeness = []
    if situation.situation: completeness.append("Situation")
    if situation.emotions: completeness.append("Emotions")
    if situation.automatic_thoughts: completeness.append("Thoughts")
    if situation.behavior: completeness.append("Behaviors")
    if situation.meaning_of_at: completeness.append("Meaning")
    
    print(f"📊 Completion: {len(completeness)}/5 areas captured")
    if completeness:
        print(f"✅ Captured: {', '.join(completeness)}")
    
    missing = []
    if not situation.situation: missing.append("Situation")
    if not situation.emotions: missing.append("Emotions")
    if not situation.automatic_thoughts: missing.append("Thoughts")
    if not situation.behavior: missing.append("Behaviors")
    if not situation.meaning_of_at: missing.append("Meaning")
    
    if missing:
        print(f"⚠️  Still needed: {', '.join(missing)}")
    
    print("="*60)