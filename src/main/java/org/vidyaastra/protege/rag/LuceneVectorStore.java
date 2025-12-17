package org.vidyaastra.protege.rag;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * File-based vector store using Apache Lucene KnnFloatVectorField
 */
public class LuceneVectorStore implements Closeable {
    
    private static final Logger logger = LoggerFactory.getLogger(LuceneVectorStore.class);
    
    private static final String ID_FIELD = "id";
    private static final String TEXT_FIELD = "text";
    private static final String VECTOR_FIELD = "vector";
    private static final String METADATA_PREFIX = "meta_";
    
    private final Directory directory;
    private final StandardAnalyzer analyzer;
    private final IndexWriter writer;
    private final int dimension;
    private final String indexPath;
    
    /**
     * Create or open a Lucene vector store
     * @param indexPath Path to index directory
     */
    public LuceneVectorStore(String indexPath) throws IOException {
        this(indexPath, 1024); // Max dimension for Lucene 9.8+
    }
    
    /**
     * Create or open a Lucene vector store with custom dimension
     * @param indexPath Path to index directory
     * @param dimension Vector dimension
     */
    public LuceneVectorStore(String indexPath, int dimension) throws IOException {
        if (dimension > 1024) {
            throw new IllegalArgumentException("Vector dimension cannot exceed 1024 (Lucene limit). Got: " + dimension);
        }
        this.indexPath = indexPath;
        this.dimension = dimension;
        this.directory = FSDirectory.open(Paths.get(indexPath));
        this.analyzer = new StandardAnalyzer();
        
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);
        this.writer = new IndexWriter(directory, config);
        
        logger.info("Initialized Lucene vector store at: {} (dimension: {})", indexPath, dimension);
    }
    
    /**
     * Add or update vectors in the store
     */
    public void upsert(List<VectorData> data) throws IOException {
        for (VectorData vectorData : data) {
            Document doc = new Document();
            
            // Store ID
            doc.add(new StringField(ID_FIELD, vectorData.getId(), Field.Store.YES));
            
            // Store text content
            doc.add(new StoredField(TEXT_FIELD, vectorData.getText()));
            doc.add(new TextField(TEXT_FIELD + "_searchable", vectorData.getText(), Field.Store.NO));
            
            // Store vector
            float[] vectorArray = toFloatArray(vectorData.getVector());
            if (vectorArray.length > 1024) {
                throw new IllegalArgumentException("Vector dimension " + vectorArray.length + " exceeds Lucene limit of 1024");
            }
            if (vectorArray.length != dimension) {
                logger.warn("Vector dimension mismatch: expected {}, got {}. Adjusting...", dimension, vectorArray.length);
                vectorArray = adjustVectorDimension(vectorArray, dimension);
            }
            doc.add(new KnnFloatVectorField(VECTOR_FIELD, vectorArray, VectorSimilarityFunction.COSINE));
            
            // Store metadata
            for (Map.Entry<String, String> entry : vectorData.getMetadata().entrySet()) {
                doc.add(new StoredField(METADATA_PREFIX + entry.getKey(), entry.getValue()));
            }
            
            // Update or insert
            writer.updateDocument(new Term(ID_FIELD, vectorData.getId()), doc);
        }
        
        writer.commit();
        logger.info("Upserted {} vectors to Lucene index", data.size());
    }
    
    /**
     * Search for similar vectors
     */
    public List<SearchResult> search(List<Float> queryVector, int topK) throws IOException {
        List<SearchResult> results = new ArrayList<>();
        
        try (IndexReader reader = DirectoryReader.open(directory)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            
            float[] queryArray = toFloatArray(queryVector);
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, queryArray, topK);
            
            TopDocs topDocs = searcher.search(query, topK);
            
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                Document doc = searcher.storedFields().document(scoreDoc.doc);
                
                String id = doc.get(ID_FIELD);
                String text = doc.get(TEXT_FIELD);
                float score = scoreDoc.score;
                
                // Extract metadata
                Map<String, String> metadata = new HashMap<>();
                for (IndexableField field : doc.getFields()) {
                    if (field.name().startsWith(METADATA_PREFIX)) {
                        String key = field.name().substring(METADATA_PREFIX.length());
                        metadata.put(key, field.stringValue());
                    }
                }
                
                results.add(new SearchResult(id, text, score, metadata));
            }
        }
        
        logger.debug("Found {} search results", results.size());
        return results;
    }
    
    /**
     * Get index statistics
     */
    public IndexStats getStats() throws IOException {
        try (IndexReader reader = DirectoryReader.open(directory)) {
            return new IndexStats(reader.numDocs(), dimension, indexPath);
        } catch (IndexNotFoundException e) {
            return new IndexStats(0, dimension, indexPath);
        }
    }
    
    /**
     * Clear all documents from index
     */
    public void clearIndex() throws IOException {
        writer.deleteAll();
        writer.commit();
        logger.info("Cleared all documents from index");
    }
    
    @Override
    public void close() throws IOException {
        if (writer != null) {
            writer.close();
        }
        if (directory != null) {
            directory.close();
        }
        logger.info("Closed Lucene vector store");
    }
    
    private float[] toFloatArray(List<Float> list) {
        float[] array = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
        return array;
    }
    
    /**
     * Adjust vector dimension by truncating or padding
     */
    private float[] adjustVectorDimension(float[] vector, int targetDimension) {
        if (vector.length == targetDimension) {
            return vector;
        }
        
        float[] adjusted = new float[targetDimension];
        if (vector.length > targetDimension) {
            // Truncate
            System.arraycopy(vector, 0, adjusted, 0, targetDimension);
            logger.debug("Truncated vector from {} to {} dimensions", vector.length, targetDimension);
        } else {
            // Pad with zeros
            System.arraycopy(vector, 0, adjusted, 0, vector.length);
            logger.debug("Padded vector from {} to {} dimensions", vector.length, targetDimension);
        }
        return adjusted;
    }
    
    /**
     * Vector data to be stored
     */
    public static class VectorData {
        private final String id;
        private final List<Float> vector;
        private final String text;
        private final Map<String, String> metadata;
        
        public VectorData(String id, List<Float> vector, String text, Map<String, String> metadata) {
            this.id = id;
            this.vector = vector;
            this.text = text;
            this.metadata = metadata != null ? metadata : new HashMap<>();
        }
        
        public String getId() {
            return id;
        }
        
        public List<Float> getVector() {
            return vector;
        }
        
        public String getText() {
            return text;
        }
        
        public Map<String, String> getMetadata() {
            return metadata;
        }
    }
    
    /**
     * Search result with score
     */
    public static class SearchResult {
        private final String id;
        private final String text;
        private final float score;
        private final Map<String, String> metadata;
        
        public SearchResult(String id, String text, float score, Map<String, String> metadata) {
            this.id = id;
            this.text = text;
            this.score = score;
            this.metadata = metadata;
        }
        
        public String getId() {
            return id;
        }
        
        public String getText() {
            return text;
        }
        
        public float getScore() {
            return score;
        }
        
        public Map<String, String> getMetadata() {
            return metadata;
        }
    }
    
    /**
     * Index statistics
     */
    public static class IndexStats {
        private final int documentCount;
        private final int vectorDimension;
        private final String indexPath;
        
        public IndexStats(int documentCount, int vectorDimension, String indexPath) {
            this.documentCount = documentCount;
            this.vectorDimension = vectorDimension;
            this.indexPath = indexPath;
        }
        
        public int getDocumentCount() {
            return documentCount;
        }
        
        public int getVectorDimension() {
            return vectorDimension;
        }
        
        public String getIndexPath() {
            return indexPath;
        }
    }
}
