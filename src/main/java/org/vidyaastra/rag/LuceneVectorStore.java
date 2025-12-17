package org.vidyaastra.rag;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Lucene-based vector store using KnnFloatVectorField for similarity search.
 * Stores vectors and metadata in file-based Lucene index.
 */
public class LuceneVectorStore implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(LuceneVectorStore.class);
    private static final String VECTOR_FIELD = "vector";
    private static final String ID_FIELD = "id";
    private static final String TEXT_FIELD = "text";
    private static final String METADATA_PREFIX = "meta_";

    private final String indexPath;
    private final int vectorDimension;
    private Directory directory;
    private IndexWriter indexWriter;

    /**
     * Create a Lucene vector store
     *
     * @param indexPath       Directory path where Lucene index will be stored
     * @param vectorDimension Dimension of vectors (e.g., 1536 for OpenAI embeddings)
     */
    public LuceneVectorStore(String indexPath, int vectorDimension) throws IOException {
        this.indexPath = indexPath;
        this.vectorDimension = vectorDimension;
        initializeIndex();
        logger.info("📁 Lucene vector store initialized at: {}", indexPath);
    }

    /**
     * Initialize or open existing Lucene index
     */
    private void initializeIndex() throws IOException {
        Path path = Paths.get(indexPath);
        this.directory = FSDirectory.open(path);

        IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);
        this.indexWriter = new IndexWriter(directory, config);

        logger.info("✅ Lucene index opened/created with {} dimensions", vectorDimension);
    }

    /**
     * Add or update vectors in the index
     */
    public void upsert(List<VectorData> vectorDataList) throws IOException {
        for (VectorData data : vectorDataList) {
            Document doc = new Document();

            // Store ID
            doc.add(new StringField(ID_FIELD, data.getId(), Field.Store.YES));

            // Store vector for KNN search
            float[] vectorArray = toFloatArray(data.getVector());
            doc.add(new KnnFloatVectorField(VECTOR_FIELD, vectorArray, VectorSimilarityFunction.COSINE));

            // Store metadata
            if (data.getMetadata() != null) {
                for (Map.Entry<String, Object> entry : data.getMetadata().entrySet()) {
                    String value = entry.getValue() != null ? entry.getValue().toString() : "";
                    doc.add(new StoredField(METADATA_PREFIX + entry.getKey(), value));

                    // Store text field separately for easy retrieval
                    if ("text".equals(entry.getKey())) {
                        doc.add(new StoredField(TEXT_FIELD, value));
                    }
                }
            }

            indexWriter.addDocument(doc);
        }

        indexWriter.commit();
        logger.info("✅ Indexed {} vectors", vectorDataList.size());
    }

    /**
     * Search for similar vectors using KNN
     */
    public List<SearchResult> search(List<Float> queryVector, int limit) throws IOException {
        List<SearchResult> results = new ArrayList<>();

        // Refresh index reader
        try (IndexReader reader = DirectoryReader.open(directory)) {
            IndexSearcher searcher = new IndexSearcher(reader);

            // Create KNN query
            float[] queryArray = toFloatArray(queryVector);
            KnnFloatVectorQuery knnQuery = new KnnFloatVectorQuery(VECTOR_FIELD, queryArray, limit);

            // Execute search
            TopDocs topDocs = searcher.search(knnQuery, limit);

            // Process results
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                Document doc = searcher.storedFields().document(scoreDoc.doc);

                // Extract metadata
                Map<String, Object> metadata = new HashMap<>();
                doc.getFields().forEach(field -> {
                    String name = field.name();
                    if (name.startsWith(METADATA_PREFIX)) {
                        String key = name.substring(METADATA_PREFIX.length());
                        metadata.put(key, field.stringValue());
                    }
                });

                results.add(new SearchResult(scoreDoc.score, metadata));
            }
        }

        logger.info("🔍 Found {} similar vectors", results.size());
        return results;
    }

    /**
     * Get index statistics
     */
    public IndexStats getStats() throws IOException {
        try (IndexReader reader = DirectoryReader.open(directory)) {
            return new IndexStats(
                    reader.numDocs(),
                    vectorDimension,
                    indexPath
            );
        }
    }

    /**
     * Delete all documents from index
     */
    public void clearIndex() throws IOException {
        indexWriter.deleteAll();
        indexWriter.commit();
        logger.info("🗑️ Index cleared");
    }

    /**
     * Convert List<Float> to float[]
     */
    private float[] toFloatArray(List<Float> list) {
        float[] array = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
        return array;
    }

    @Override
    public void close() throws IOException {
        if (indexWriter != null) {
            indexWriter.close();
        }
        if (directory != null) {
            directory.close();
        }
        logger.info("✅ Lucene vector store closed");
    }

    /**
     * Vector data container
     */
    public static class VectorData {
        private final String id;
        private final List<Float> vector;
        private final Map<String, Object> metadata;

        public VectorData(String id, List<Float> vector, Map<String, Object> metadata) {
            this.id = id;
            this.vector = vector;
            this.metadata = metadata;
        }

        public String getId() {
            return id;
        }

        public List<Float> getVector() {
            return vector;
        }

        public Map<String, Object> getMetadata() {
            return metadata;
        }
    }

    /**
     * Search result container
     */
    public static class SearchResult {
        private final float score;
        private final Map<String, Object> metadata;

        public SearchResult(float score, Map<String, Object> metadata) {
            this.score = score;
            this.metadata = metadata;
        }

        public float getScore() {
            return score;
        }

        public Map<String, Object> getMetadata() {
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
