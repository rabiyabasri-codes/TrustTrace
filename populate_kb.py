import sys, os
sys.path.append('C:/Users/Rabiya Basri/OneDrive/Desktop/TrustTrace')
from victim_pipeline.agents import knowledge_base, reset_knowledge_base_for_benign

reset_knowledge_base_for_benign(knowledge_base)
print('Knowledge base populated with reference docs')
