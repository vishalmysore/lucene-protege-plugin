package org.vidyaastra.protege.rag;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import io.github.vishalmysore.graph.*;
import okhttp3.*;
import org.semanticweb.owlapi.model.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * RAG Service orchestrating Neo4j, Lucene Vector Store, and AI for queries
 */
public class RagService implements AutoCloseable {
    
    private static final Logger logger = LoggerFactory.getLogger(RagService.class);
    
    private final Neo4jService neo4jService;
    private final LuceneVectorStore vectorStore;
    private final EmbeddingService embeddingService;
    private final OkHttpClient httpClient;
    private final Gson gson;
    private final String aiModel;
    private final String aiApiKey;
    private String chunkingStrategy;  // Not final - can be updated
    private OWLOntology ontology;
    
    public RagService(String neo4jUri, String neo4jUsername, String neo4jPassword, 
                     String neo4jDatabase, String luceneIndexPath,
                     String embeddingModel, String embeddingApiKey,
                     String aiModel, String aiApiKey, String chunkingStrategy) throws IOException {
        
        this.neo4jService = new Neo4jService(neo4jUri, neo4jUsername, neo4jPassword, neo4jDatabase);
        this.embeddingService = new EmbeddingService(embeddingModel, embeddingApiKey);
        
        // Create vector store with the embedding model's dimension
        int dimension = embeddingService.getDimension();
        this.vectorStore = new LuceneVectorStore(luceneIndexPath, dimension);
        
        this.aiModel = extractModelName(aiModel);
        this.aiApiKey = aiApiKey;
        this.chunkingStrategy = chunkingStrategy;
        
        this.httpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .build();
        this.gson = new Gson();
        
        logger.info("Initialized RagService with Lucene index at: {} and chunking strategy: {}", luceneIndexPath, chunkingStrategy);
    }
    
    /**
     * Set the OWL ontology to be indexed
     */
    public void setOntology(OWLOntology ontology) {
        this.ontology = ontology;
    }
    
    /**
     * Set the chunking strategy
     */
    public void setChunkingStrategy(String strategy) {
        this.chunkingStrategy = strategy;
        logger.info("Chunking strategy updated to: {}", strategy);
    }
    
    /**
     * Apply text chunking based on selected strategy
     */
    private List<String> applyChunking(String text) {
        List<String> chunks = new ArrayList<>();
        
        switch (chunkingStrategy) {
            case "SentenceChunking":
                // Split by sentences
                String[] sentences = text.split("(?<=[.!?])\\s+");
                chunks.addAll(Arrays.asList(sentences));
                break;
            case "ParagraphChunking":
                // Split by paragraphs
                String[] paragraphs = text.split("\\n\\s*\\n");
                chunks.addAll(Arrays.asList(paragraphs));
                break;
            case "FixedSizeChunking":
                // Fixed size chunks (500 chars)
                int chunkSize = 500;
                for (int i = 0; i < text.length(); i += chunkSize) {
                    chunks.add(text.substring(i, Math.min(i + chunkSize, text.length())));
                }
                break;
            case "WordChunking":
            default:
                // Split by word count (100 words per chunk)
                String[] words = text.split("\\s+");
                int wordsPerChunk = 100;
                for (int i = 0; i < words.length; i += wordsPerChunk) {
                    int end = Math.min(i + wordsPerChunk, words.length);
                    chunks.add(String.join(" ", Arrays.copyOfRange(words, i, end)));
                }
                break;
        }
        
        return chunks.isEmpty() ? Collections.singletonList(text) : chunks;
    }
    
    private String extractModelName(String selection) {
        if (selection.contains("(")) {
            return selection.substring(0, selection.indexOf("(")).trim();
        }
        return selection;
    }
    
    /**
     * Execute a RAG query combining vector search and AI generation
     */
    public String executeRagQuery(String userQuery, int topK) throws IOException {
        logger.info("Executing RAG query: {} (topK={})", userQuery, topK);
        
        // Generate embedding for the query
        List<Float> queryEmbedding = embeddingService.generateEmbedding(userQuery);
        
        // Search vector store for relevant chunks
        List<LuceneVectorStore.SearchResult> searchResults = vectorStore.search(queryEmbedding, topK);
        
        if (searchResults.isEmpty()) {
            return "No relevant information found in the knowledge base.";
        }
        
        // Build similarity scores summary
        StringBuilder scoresSummary = new StringBuilder();
        scoresSummary.append("=== SIMILARITY SCORES (Chunking: ").append(chunkingStrategy).append(") ===\n");
        for (int i = 0; i < searchResults.size(); i++) {
            LuceneVectorStore.SearchResult result = searchResults.get(i);
            scoresSummary.append("Result ").append(i + 1)
                        .append(": ").append(String.format("%.4f", result.getScore()))
                        .append("\n");
        }
        scoresSummary.append("\n");
        
        // Build context from search results
        StringBuilder context = new StringBuilder();
        for (int i = 0; i < searchResults.size(); i++) {
            LuceneVectorStore.SearchResult result = searchResults.get(i);
            context.append("--- Context ").append(i + 1)
                   .append(" (Similarity: ").append(String.format("%.4f", result.getScore()))
                   .append(") ---\n")
                   .append(result.getText())
                   .append("\n\n");
        }
        
        // Execute Cypher query if user query looks like a graph query
        String cypherContext = "";
        if (shouldExecuteCypherQuery(userQuery)) {
            try {
                String generatedCypher = generateCypherQuery(userQuery);
                logger.info("Generated Cypher: {}", generatedCypher);
                
                List<Map<String, Object>> cypherResults = neo4jService.executeQuery(generatedCypher);
                cypherContext = "\n--- Neo4j Query Results ---\n" + formatQueryResults(cypherResults) + "\n\n";
            } catch (Exception e) {
                logger.warn("Failed to execute Cypher query: {}", e.getMessage());
                cypherContext = "\n[Note: Graph query execution failed]\n\n";
            }
        }
        
        // Generate answer using AI
        return scoresSummary.toString() + context.toString() + cypherContext + "\n=== AI RESPONSE ===\n" + generateAiResponse(userQuery, context.toString() + cypherContext);
    }
    
    /**
     * Index graph data to vector store with text chunking
     */
    public int indexGraphToVectorStore() throws IOException {
        logger.info("Starting graph to vector store indexing with chunking strategy: {}", chunkingStrategy);
        
        List<Neo4jService.GraphChunk> chunks = neo4jService.getGraphChunks();
        if (chunks.isEmpty()) {
            logger.warn("No graph chunks to index");
            return 0;
        }
        
        List<LuceneVectorStore.VectorData> vectorDataList = new ArrayList<>();
        
        for (Neo4jService.GraphChunk chunk : chunks) {
            // Apply text chunking strategy
            List<String> subChunks = applyChunking(chunk.getText());
            
            // Process each sub-chunk
            for (int i = 0; i < subChunks.size(); i++) {
                String subChunkText = subChunks.get(i).trim();
                if (subChunkText.isEmpty()) continue;
                
                List<Float> embedding = embeddingService.generateEmbedding(subChunkText);
                
                Map<String, String> metadata = new HashMap<>();
                metadata.put("source", "neo4j");
                metadata.put("type", "graph_chunk");
                metadata.put("chunking_strategy", chunkingStrategy);
                metadata.put("parent_chunk_id", chunk.getText().hashCode() + "");
                metadata.put("sub_chunk_index", String.valueOf(i));
                
                // Add node properties to metadata
                for (Map.Entry<String, Object> entry : chunk.getMetadata().entrySet()) {
                    metadata.put(entry.getKey(), String.valueOf(entry.getValue()));
                }
                
                vectorDataList.add(new LuceneVectorStore.VectorData(
                    UUID.randomUUID().toString(),
                    embedding,
                    subChunkText,
                    metadata
                ));
            }
        }
        
        vectorStore.upsert(vectorDataList);
        logger.info("Indexed {} chunks (using {}) to vector store", vectorDataList.size(), chunkingStrategy);
        
        return vectorDataList.size();
    }
    
    /**
     * Index OWL ontology to vector store
     */
    public int indexOntologyToVectorStore() throws IOException {
        if (ontology == null) {
            logger.warn("No ontology loaded");
            return 0;
        }
        
        logger.info("Starting ontology indexing with chunking strategy: {}", chunkingStrategy);
        
        // Check if using OWL-specific chunker from agenticmemory
        if (isOWLChunkingStrategy(chunkingStrategy)) {
            return indexWithOWLChunker();
        }
        
        // Otherwise use text-based chunking
        List<String> ontologyTexts = extractOntologyTexts();
        if (ontologyTexts.isEmpty()) {
            logger.warn("No ontology content to index");
            return 0;
        }
        
        List<LuceneVectorStore.VectorData> vectorDataList = new ArrayList<>();
        
        for (String text : ontologyTexts) {
            // Apply text chunking strategy
            List<String> subChunks = applyChunking(text);
            
            // Process each sub-chunk
            for (int i = 0; i < subChunks.size(); i++) {
                String subChunkText = subChunks.get(i).trim();
                if (subChunkText.isEmpty()) continue;
                
                List<Float> embedding = embeddingService.generateEmbedding(subChunkText);
                
                Map<String, String> metadata = new HashMap<>();
                metadata.put("source", "ontology");
                metadata.put("type", "owl_chunk");
                metadata.put("chunking_strategy", chunkingStrategy);
                metadata.put("parent_chunk_id", text.hashCode() + "");
                metadata.put("sub_chunk_index", String.valueOf(i));
                
                vectorDataList.add(new LuceneVectorStore.VectorData(
                    UUID.randomUUID().toString(),
                    embedding,
                    subChunkText,
                    metadata
                ));
            }
        }
        
        vectorStore.upsert(vectorDataList);
        logger.info("Indexed {} ontology chunks (using {}) to vector store", vectorDataList.size(), chunkingStrategy);
        
        return vectorDataList.size();
    }
    
    /**
     * Extract text content from OWL ontology
     */
    private List<String> extractOntologyTexts() {
        List<String> texts = new ArrayList<>();
        
        // Extract classes with their annotations and axioms
        for (OWLClass cls : ontology.getClassesInSignature()) {
            StringBuilder sb = new StringBuilder();
            sb.append("Class: ").append(getEntityLabel(cls)).append("\n");
            sb.append("IRI: ").append(cls.getIRI()).append("\n");
            
            // Add annotations
            for (OWLAnnotationAssertionAxiom axiom : ontology.getAnnotationAssertionAxioms(cls.getIRI())) {
                sb.append(getAnnotationText(axiom.getAnnotation())).append("\n");
            }
            
            // Add superclasses
            for (OWLSubClassOfAxiom axiom : ontology.getSubClassAxiomsForSubClass(cls)) {
                sb.append("SuperClass: ").append(axiom.getSuperClass()).append("\n");
            }
            
            texts.add(sb.toString());
        }
        
        // Extract object properties
        for (OWLObjectProperty prop : ontology.getObjectPropertiesInSignature()) {
            StringBuilder sb = new StringBuilder();
            sb.append("ObjectProperty: ").append(getEntityLabel(prop)).append("\n");
            sb.append("IRI: ").append(prop.getIRI()).append("\n");
            
            for (OWLAnnotationAssertionAxiom axiom : ontology.getAnnotationAssertionAxioms(prop.getIRI())) {
                sb.append(getAnnotationText(axiom.getAnnotation())).append("\n");
            }
            
            // Add domain and range
            for (OWLObjectPropertyDomainAxiom axiom : ontology.getObjectPropertyDomainAxioms(prop)) {
                sb.append("Domain: ").append(axiom.getDomain()).append("\n");
            }
            for (OWLObjectPropertyRangeAxiom axiom : ontology.getObjectPropertyRangeAxioms(prop)) {
                sb.append("Range: ").append(axiom.getRange()).append("\n");
            }
            
            texts.add(sb.toString());
        }
        
        // Extract data properties
        for (OWLDataProperty prop : ontology.getDataPropertiesInSignature()) {
            StringBuilder sb = new StringBuilder();
            sb.append("DataProperty: ").append(getEntityLabel(prop)).append("\n");
            sb.append("IRI: ").append(prop.getIRI()).append("\n");
            
            for (OWLAnnotationAssertionAxiom axiom : ontology.getAnnotationAssertionAxioms(prop.getIRI())) {
                sb.append(getAnnotationText(axiom.getAnnotation())).append("\n");
            }
            
            texts.add(sb.toString());
        }
        
        // Extract individuals
        for (OWLNamedIndividual individual : ontology.getIndividualsInSignature()) {
            StringBuilder sb = new StringBuilder();
            sb.append("Individual: ").append(getEntityLabel(individual)).append("\n");
            sb.append("IRI: ").append(individual.getIRI()).append("\n");
            
            for (OWLAnnotationAssertionAxiom axiom : ontology.getAnnotationAssertionAxioms(individual.getIRI())) {
                sb.append(getAnnotationText(axiom.getAnnotation())).append("\n");
            }
            
            // Add class assertions (types)
            for (OWLClassAssertionAxiom axiom : ontology.getClassAssertionAxioms(individual)) {
                OWLClassExpression classExpr = axiom.getClassExpression();
                if (!classExpr.isAnonymous()) {
                    sb.append("Type: ").append(getEntityLabel(classExpr.asOWLClass())).append("\n");
                }
            }
            
            // Add data property assertions (e.g., caseNumber, filingDate, caseStatus)
            for (OWLDataPropertyAssertionAxiom axiom : ontology.getDataPropertyAssertionAxioms(individual)) {
                OWLDataProperty prop = axiom.getProperty().asOWLDataProperty();
                OWLLiteral value = axiom.getObject();
                sb.append(getEntityLabel(prop)).append(": ").append(value.getLiteral()).append("\n");
            }
            
            // Add object property assertions (e.g., representedBy, presides, filedIn)
            for (OWLObjectPropertyAssertionAxiom axiom : ontology.getObjectPropertyAssertionAxioms(individual)) {
                if (!axiom.getProperty().isAnonymous() && axiom.getObject().isNamed()) {
                    OWLObjectProperty prop = axiom.getProperty().asOWLObjectProperty();
                    OWLNamedIndividual object = axiom.getObject().asOWLNamedIndividual();
                    sb.append(getEntityLabel(prop)).append(": ").append(getEntityLabel(object)).append("\n");
                }
            }
            
            texts.add(sb.toString());
        }
        
        logger.info("Extracted {} text chunks from ontology", texts.size());
        return texts;
    }
    
    /**
     * Get human-readable label for an entity
     */
    private String getEntityLabel(OWLEntity entity) {
        for (OWLAnnotationAssertionAxiom axiom : ontology.getAnnotationAssertionAxioms(entity.getIRI())) {
            if (axiom.getProperty().isLabel()) {
                OWLAnnotationValue value = axiom.getValue();
                if (value instanceof OWLLiteral) {
                    return ((OWLLiteral) value).getLiteral();
                }
            }
        }
        return entity.getIRI().getShortForm();
    }
    
    /**
     * Convert annotation to text
     */
    private String getAnnotationText(OWLAnnotation annotation) {
        String property = annotation.getProperty().getIRI().getShortForm();
        OWLAnnotationValue value = annotation.getValue();
        
        if (value instanceof OWLLiteral) {
            return property + ": " + ((OWLLiteral) value).getLiteral();
        } else if (value instanceof IRI) {
            return property + ": " + ((IRI) value).getShortForm();
        }
        return property + ": " + value.toString();
    }
    
    /**
     * Get vector store statistics
     */
    public LuceneVectorStore.IndexStats getVectorStoreStats() throws IOException {
        return vectorStore.getStats();
    }
    
    /**
     * Clear vector store
     */
    public void clearVectorStore() throws IOException {
        vectorStore.clearIndex();
        logger.info("Vector store cleared");
    }
    
    /**
     * Get graph schema
     */
    public String getGraphSchema() {
        return neo4jService.getGraphSchema();
    }
    
    private boolean shouldExecuteCypherQuery(String query) {
        String lowerQuery = query.toLowerCase();
        return lowerQuery.contains("find") || lowerQuery.contains("show") || 
               lowerQuery.contains("list") || lowerQuery.contains("get") ||
               lowerQuery.contains("how many") || lowerQuery.contains("count");
    }
    
    private String generateCypherQuery(String userQuery) throws IOException {
        String schema = neo4jService.getGraphSchema();
        
        String prompt = String.format(
            "Given this Neo4j graph schema:\n%s\n\n" +
            "Generate a Cypher query to answer: %s\n\n" +
            "Return ONLY the Cypher query without explanation.",
            schema, userQuery
        );
        
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", aiModel);
        
        JsonObject message = new JsonObject();
        message.addProperty("role", "user");
        message.addProperty("content", prompt);
        
        com.google.gson.JsonArray messages = new com.google.gson.JsonArray();
        messages.add(message);
        requestBody.add("messages", messages);
        
        Request request = new Request.Builder()
            .url("https://api.openai.com/v1/chat/completions")
            .addHeader("Authorization", "Bearer " + aiApiKey)
            .addHeader("Content-Type", "application/json")
            .post(RequestBody.create(
                gson.toJson(requestBody),
                MediaType.get("application/json")
            ))
            .build();
        
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("AI request failed: " + response.code());
            }
            
            JsonObject responseJson = gson.fromJson(response.body().string(), JsonObject.class);
            return responseJson
                .getAsJsonArray("choices")
                .get(0).getAsJsonObject()
                .getAsJsonObject("message")
                .get("content").getAsString()
                .trim()
                .replaceAll("```cypher", "")
                .replaceAll("```", "")
                .trim();
        }
    }
    
    private String generateAiResponse(String query, String context) throws IOException {
        String prompt = String.format(
            "Answer the following question based on the provided context. " +
            "If the context doesn't contain enough information, say so.\n\n" +
            "Context:\n%s\n\n" +
            "Question: %s\n\n" +
            "Answer:",
            context, query
        );
        
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", aiModel);
        
        JsonObject message = new JsonObject();
        message.addProperty("role", "user");
        message.addProperty("content", prompt);
        
        com.google.gson.JsonArray messages = new com.google.gson.JsonArray();
        messages.add(message);
        requestBody.add("messages", messages);
        
        Request request = new Request.Builder()
            .url("https://api.openai.com/v1/chat/completions")
            .addHeader("Authorization", "Bearer " + aiApiKey)
            .addHeader("Content-Type", "application/json")
            .post(RequestBody.create(
                gson.toJson(requestBody),
                MediaType.get("application/json")
            ))
            .build();
        
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("AI request failed: " + response.code());
            }
            
            JsonObject responseJson = gson.fromJson(response.body().string(), JsonObject.class);
            return responseJson
                .getAsJsonArray("choices")
                .get(0).getAsJsonObject()
                .getAsJsonObject("message")
                .get("content").getAsString();
        }
    }
    
    private String formatQueryResults(List<Map<String, Object>> results) {
        if (results.isEmpty()) {
            return "No results found.";
        }
        
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < results.size(); i++) {
            sb.append("Result ").append(i + 1).append(": ").append(results.get(i)).append("\n");
        }
        return sb.toString();
    }
    
    /**
     * Check if the chunking strategy is an OWL-specific one from agenticmemory
     */
    private boolean isOWLChunkingStrategy(String strategy) {
        return strategy.equals("ClassBasedChunking") ||
               strategy.equals("AnnotationBasedChunking") ||
               strategy.equals("NamespaceBasedChunking") ||
               strategy.equals("DepthBasedChunking") ||
               strategy.equals("ModuleExtractionChunking") ||
               strategy.equals("SizeBasedChunking");
    }
    
    /**
     * Index ontology using OWL-aware chunkers from agenticmemory
     */
    private int indexWithOWLChunker() throws IOException {
        logger.info("Using OWL-aware chunker: {}", chunkingStrategy);
        
        List<OWLChunk> owlChunks;
        
        try {
            switch (chunkingStrategy) {
                case "ClassBasedChunking":
                    owlChunks = new ClassBasedChunker().chunk(ontology);
                    break;
                case "AnnotationBasedChunking":
                    owlChunks = new AnnotationBasedChunker().chunk(ontology);
                    break;
                case "NamespaceBasedChunker":
                    owlChunks = new NamespaceBasedChunker().chunk(ontology);
                    break;
                case "DepthBasedChunking":
                    owlChunks = new DepthBasedChunker().chunk(ontology);
                    break;
                case "ModuleExtractionChunking":
                    owlChunks = new ModuleExtractionChunker().chunk(ontology);
                    break;
                case "SizeBasedChunking":
                    owlChunks = new SizeBasedChunker(50).chunk(ontology);  // max 50 axioms per chunk
                    break;
                default:
                    logger.warn("Unknown OWL chunking strategy: {}, falling back to ClassBasedChunker", chunkingStrategy);
                    owlChunks = new ClassBasedChunker().chunk(ontology);
            }
        } catch (Exception e) {
            logger.error("Error during OWL chunking", e);
            throw new IOException("OWL chunking failed: " + e.getMessage(), e);
        }
        
        logger.info("OWL chunker created {} chunks", owlChunks.size());
        
        List<LuceneVectorStore.VectorData> vectorDataList = new ArrayList<>();
        
        for (OWLChunk chunk : owlChunks) {
            // Convert OWL chunk to text representation
            StringBuilder text = new StringBuilder();
            text.append("Chunk ID: ").append(chunk.getId()).append("\n");
            text.append("Strategy: ").append(chunk.getStrategy()).append("\n");
            text.append("Axiom Count: ").append(chunk.getAxiomCount()).append("\n");
            
            // Add metadata
            for (Map.Entry<String, Object> entry : chunk.getMetadata().entrySet()) {
                text.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
            text.append("\n");
            
            // Add axioms
            text.append(chunk.toOWLString());
            
            String chunkText = text.toString();
            if (chunkText.trim().isEmpty()) continue;
            
            List<Float> embedding = embeddingService.generateEmbedding(chunkText);
            
            Map<String, String> metadata = new HashMap<>();
            metadata.put("source", "ontology");
            metadata.put("type", "owl_chunk");
            metadata.put("chunking_strategy", chunkingStrategy);
            metadata.put("chunk_id", chunk.getId());
            metadata.put("axiom_count", String.valueOf(chunk.getAxiomCount()));
            metadata.put("strategy_used", chunk.getStrategy().toString());
            
            vectorDataList.add(new LuceneVectorStore.VectorData(
                UUID.randomUUID().toString(),
                embedding,
                chunkText,
                metadata
            ));
        }
        
        vectorStore.upsert(vectorDataList);
        logger.info("Indexed {} OWL chunks to vector store", vectorDataList.size());
        
        return vectorDataList.size();
    }
    
    @Override
    public void close() {
        try {
            if (vectorStore != null) {
                vectorStore.close();
            }
        } catch (IOException e) {
            logger.error("Error closing vector store", e);
        }
        
        if (neo4jService != null) {
            neo4jService.close();
        }
        
        logger.info("RagService closed");
    }
}
