import yaml
from src.document_processing.document_processor import MultimodalDocumentProcessor
from src.chunking.semantic_chunker import SemanticContextPreservingChunker
from src.entity_extraction.entity_extractor import EntityExtractor
from src.entity_extraction.relationship_extractor import RelationshipExtractor
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.retrieval.context_retriever import ContextAwareRetriever
from src.retrieval.context_assembler import ContextAssembler

# Load configurations
with open("configs/model_config.yaml") as f:
    model_config = yaml.safe_load(f)
with open("configs/processing_config.yaml") as f:
    processing_config = yaml.safe_load(f)

# Initialize components
processor = MultimodalDocumentProcessor(model_config)
chunker = SemanticContextPreservingChunker(processing_config)
entity_extractor = EntityExtractor(model_config)
relationship_extractor = RelationshipExtractor(processing_config)
graph_builder = KnowledgeGraphBuilder(model_config)
retriever = ContextAwareRetriever(graph_builder.build_graph([], []), processing_config)
assembler = ContextAssembler(processing_config)

# Process and index a document
document = processor.process_document("/dataset/472.pdf")
chunks = chunker.chunk_document(document)
entities = entity_extractor.extract_entities(chunks)
relationships = relationship_extractor.extract_relationships(entities, chunks)
knowledge_graph = graph_builder.build_graph(chunks, entities, relationships)
retriever = ContextAwareRetriever(knowledge_graph, processing_config)

# Process a query
query = "What are the main findings in the document?"
retrieved_chunks = retriever.retrieve(query, top_k=5)
context = assembler.assemble_context(retrieved_chunks, knowledge_graph, retriever.embedder.encode(query))

# Display results
print("Retrieved Context:")
print(context)