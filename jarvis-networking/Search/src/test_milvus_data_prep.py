import unittest
import os
import json
from milvus_data_prep import (
    load_summarized_profiles,
    generate_embeddings,
    prepare_milvus_collection,
    insert_to_milvus,
    query_milvus
)

class TestMilvusDataPrep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test data"""
        cls.test_file = "test_summarized_profiles.json"
        cls.test_data = [
            {"id": 1, "summary": "Software engineer with 5 years experience", "vector": [0.1]*768},
            {"id": 2, "summary": "Data scientist specializing in NLP", "vector": [0.2]*768}
        ]
        
        with open(cls.test_file, 'w') as f:
            for item in cls.test_data:
                json.dump(item, f)
                f.write('\n')
    
    def test_load_summarized_profiles(self):
        """Test loading summarized profiles"""
        profiles = list(load_summarized_profiles(self.test_file))
        self.assertEqual(len(profiles), 2)
        self.assertEqual(profiles[0]['id'], 1)
    
    def test_generate_embeddings(self):
        """Test embedding generation"""
        profiles = list(load_summarized_profiles(self.test_file))
        embeddings = generate_embeddings(profiles)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 768)  # Assuming 768-dim embeddings
    
    def test_milvus_operations(self):
        """Test full Milvus pipeline"""
        # Prepare test collection
        collection_name = "test_profiles"
        prepare_milvus_collection(collection_name)
        
        # Insert test data
        profiles = list(load_summarized_profiles(self.test_file))
        embeddings = generate_embeddings(profiles)
        insert_to_milvus(collection_name, profiles, embeddings)
        
        # Query test
        results = query_milvus(collection_name, "software engineer", top_k=1)
        self.assertTrue(len(results) > 0)
        
        # Cleanup
        from pymilvus import utility
        utility.drop_collection(collection_name)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)

if __name__ == '__main__':
    unittest.main()
