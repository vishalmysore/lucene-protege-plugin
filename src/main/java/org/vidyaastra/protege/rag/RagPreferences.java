package org.vidyaastra.protege.rag;

import org.protege.editor.core.prefs.Preferences;
import org.protege.editor.core.prefs.PreferencesManager;

/**
 * Manages plugin preferences for Neo4j, Lucene vector store, and AI model configurations
 */
public class RagPreferences {
    
    private static final String PREFERENCES_ID = "com.protege.lucene.rag";
    
    // Neo4j preference keys
    private static final String NEO4J_URI_KEY = "neo4j.uri";
    private static final String NEO4J_USERNAME_KEY = "neo4j.username";
    private static final String NEO4J_PASSWORD_KEY = "neo4j.password";
    private static final String NEO4J_DATABASE_KEY = "neo4j.database";
    
    // Lucene Vector Store preference keys
    private static final String LUCENE_INDEX_PATH_KEY = "lucene.index.path";
    
    // Embedding model preference keys
    private static final String EMBEDDING_MODEL_KEY = "embedding.model";
    private static final String EMBEDDING_API_KEY_KEY = "embedding.apikey";
    
    // AI model preference keys
    private static final String AI_MODEL_KEY = "ai.model";
    private static final String AI_API_KEY_KEY = "ai.apikey";
    
    // Chunking strategy preference key
    private static final String CHUNKING_STRATEGY_KEY = "chunking.strategy";
    
    // Default values
    private static final String DEFAULT_NEO4J_URI = "bolt://localhost:7687";
    private static final String DEFAULT_NEO4J_USERNAME = "neo4j";
    private static final String DEFAULT_NEO4J_DATABASE = "neo4j";
    
    private static final String DEFAULT_LUCENE_INDEX_PATH = "./lucene_index";
    
    private static final String DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small (OpenAI)";
    private static final String DEFAULT_AI_MODEL = "gpt-4o-mini (OpenAI)";
    private static final String DEFAULT_CHUNKING_STRATEGY = "WordChunking";
    
    private Preferences getPreferences() {
        return PreferencesManager.getInstance()
            .getPreferencesForSet(PREFERENCES_ID, PREFERENCES_ID);
    }
    
    // Neo4j getters and setters
    public String getNeo4jUri() {
        return getPreferences().getString(NEO4J_URI_KEY, DEFAULT_NEO4J_URI);
    }
    
    public void setNeo4jUri(String uri) {
        getPreferences().putString(NEO4J_URI_KEY, uri);
    }
    
    public String getNeo4jUsername() {
        return getPreferences().getString(NEO4J_USERNAME_KEY, DEFAULT_NEO4J_USERNAME);
    }
    
    public void setNeo4jUsername(String username) {
        getPreferences().putString(NEO4J_USERNAME_KEY, username);
    }
    
    public String getNeo4jPassword() {
        return getPreferences().getString(NEO4J_PASSWORD_KEY, "");
    }
    
    public void setNeo4jPassword(String password) {
        getPreferences().putString(NEO4J_PASSWORD_KEY, password);
    }
    
    public String getNeo4jDatabase() {
        return getPreferences().getString(NEO4J_DATABASE_KEY, DEFAULT_NEO4J_DATABASE);
    }
    
    public void setNeo4jDatabase(String database) {
        getPreferences().putString(NEO4J_DATABASE_KEY, database);
    }
    
    // Lucene Vector Store getters and setters
    public String getLuceneIndexPath() {
        return getPreferences().getString(LUCENE_INDEX_PATH_KEY, DEFAULT_LUCENE_INDEX_PATH);
    }
    
    public void setLuceneIndexPath(String path) {
        getPreferences().putString(LUCENE_INDEX_PATH_KEY, path);
    }
    
    // Embedding model getters and setters
    public String getEmbeddingModel() {
        return getPreferences().getString(EMBEDDING_MODEL_KEY, DEFAULT_EMBEDDING_MODEL);
    }
    
    public void setEmbeddingModel(String model) {
        getPreferences().putString(EMBEDDING_MODEL_KEY, model);
    }
    
    public String getEmbeddingApiKey() {
        return getPreferences().getString(EMBEDDING_API_KEY_KEY, "");
    }
    
    public void setEmbeddingApiKey(String apiKey) {
        getPreferences().putString(EMBEDDING_API_KEY_KEY, apiKey);
    }
    
    // AI model getters and setters
    public String getAiModel() {
        return getPreferences().getString(AI_MODEL_KEY, DEFAULT_AI_MODEL);
    }
    
    public void setAiModel(String model) {
        getPreferences().putString(AI_MODEL_KEY, model);
    }
    
    public String getAiApiKey() {
        return getPreferences().getString(AI_API_KEY_KEY, "");
    }
    
    public void setAiApiKey(String apiKey) {
        getPreferences().putString(AI_API_KEY_KEY, apiKey);
    }
    
    // Chunking strategy getters and setters
    public String getChunkingStrategy() {
        return getPreferences().getString(CHUNKING_STRATEGY_KEY, DEFAULT_CHUNKING_STRATEGY);
    }
    
    public void setChunkingStrategy(String strategy) {
        getPreferences().putString(CHUNKING_STRATEGY_KEY, strategy);
    }
}
