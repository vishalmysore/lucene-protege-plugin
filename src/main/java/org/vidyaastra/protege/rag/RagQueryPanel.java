package org.vidyaastra.protege.rag;

import org.protege.editor.owl.ui.view.AbstractOWLViewComponent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

/**
 * Main UI panel for Neo4j-Lucene RAG plugin
 * Allows users to configure connections and perform RAG-based queries with file-based vector storage
 */
public class RagQueryPanel extends AbstractOWLViewComponent {
    
    private static final Logger logger = LoggerFactory.getLogger(RagQueryPanel.class);
    
    // Connection Configuration Fields
    private JTextField neo4jUriField;
    private JTextField neo4jUsernameField;
    private JPasswordField neo4jPasswordField;
    private JTextField neo4jDatabaseField;
    
    private JTextField luceneIndexPathField;
    
    private JComboBox<String> embeddingModelCombo;
    private JPasswordField embeddingApiKeyField;
    
    private JComboBox<String> aiModelCombo;
    private JPasswordField aiApiKeyField;
    
    private JComboBox<String> chunkingStrategyCombo;
    
    // Status indicators
    private JLabel neo4jStatusLabel;
    private JLabel vectorStoreStatusLabel;
    
    // Query Components
    private JTextArea queryTextArea;
    private JTextArea resultsTextArea;
    private JTextArea statsArea;
    private JButton executeButton;
    private JButton saveSettingsButton;
    private JButton connectButton;
    
    // Service instances
    private RagPreferences preferences;
    private RagService ragService;
    
    @Override
    protected void initialiseOWLView() {
        setLayout(new BorderLayout());
        preferences = new RagPreferences();
        
        // Create main panel with tabs
        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.addTab("Configuration", createConfigurationPanel());
        tabbedPane.addTab("RAG Query", createQueryPanel());
        tabbedPane.addTab("Vector Store", createVectorStorePanel());
        
        add(tabbedPane, BorderLayout.CENTER);
        
        // Load saved preferences
        loadPreferences();
    }
    
    private JPanel createConfigurationPanel() {
        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        JPanel configGrid = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(5, 5, 5, 5);
        
        // Neo4j Configuration Section
        addSectionHeader(configGrid, gbc, 0, "Neo4j Configuration");
        
        gbc.gridy = 1;
        addLabeledField(configGrid, gbc, "URI:", neo4jUriField = new JTextField(30));
        neo4jUriField.setToolTipText("e.g., neo4j+s://xxxxx.databases.neo4j.io or bolt://localhost:7687");
        
        gbc.gridy = 2;
        addLabeledField(configGrid, gbc, "Username:", neo4jUsernameField = new JTextField(30));
        
        gbc.gridy = 3;
        addLabeledField(configGrid, gbc, "Password:", neo4jPasswordField = new JPasswordField(30));
        
        gbc.gridy = 4;
        addLabeledField(configGrid, gbc, "Database:", neo4jDatabaseField = new JTextField(30));
        neo4jDatabaseField.setText("neo4j");
        
        gbc.gridy = 5;
        gbc.gridx = 1;
        neo4jStatusLabel = new JLabel("⚪ Not Connected");
        configGrid.add(neo4jStatusLabel, gbc);
        
        // Lucene Vector Store Configuration Section
        gbc.gridy = 6;
        addSectionHeader(configGrid, gbc, 6, "Lucene Vector Store Configuration");
        
        gbc.gridy = 7;
        addLabeledField(configGrid, gbc, "Index Path:", luceneIndexPathField = new JTextField(30));
        luceneIndexPathField.setText("./lucene_index");
        luceneIndexPathField.setToolTipText("Local directory for Lucene index storage");
        
        gbc.gridy = 8;
        gbc.gridx = 1;
        vectorStoreStatusLabel = new JLabel("⚪ Not Initialized");
        configGrid.add(vectorStoreStatusLabel, gbc);
        
        // Embedding Model Configuration
        gbc.gridy = 9;
        addSectionHeader(configGrid, gbc, 9, "Embedding Model");
        
        gbc.gridy = 10;
        String[] embeddingModels = {
            "text-embedding-3-small (OpenAI)",
            "text-embedding-3-large (OpenAI)",
            "text-embedding-ada-002 (OpenAI)",
            "embed-english-v3.0 (Cohere)",
            "all-MiniLM-L6-v2 (Local)",
            "all-mpnet-base-v2 (Local)"
        };
        addLabeledField(configGrid, gbc, "Model:", embeddingModelCombo = new JComboBox<>(embeddingModels));
        
        gbc.gridy = 11;
        addLabeledField(configGrid, gbc, "API Key:", embeddingApiKeyField = new JPasswordField(30));
        embeddingApiKeyField.setToolTipText("Leave empty for local models");
        
        // AI Model Configuration
        gbc.gridy = 12;
        addSectionHeader(configGrid, gbc, 12, "AI Model (for RAG)");
        
        gbc.gridy = 13;
        String[] aiModels = {
            "gpt-4o (OpenAI)",
            "gpt-4o-mini (OpenAI)",
            "claude-3-5-sonnet (Anthropic)",
            "claude-3-opus (Anthropic)",
            "llama3:8b (Ollama Local)",
            "mistral (Ollama Local)"
        };
        addLabeledField(configGrid, gbc, "Model:", aiModelCombo = new JComboBox<>(aiModels));
        
        gbc.gridy = 14;
        addLabeledField(configGrid, gbc, "API Key:", aiApiKeyField = new JPasswordField(30));
        aiApiKeyField.setToolTipText("Leave empty for Ollama local models");
        

        // Buttons
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        saveSettingsButton = new JButton("Save Settings");
        saveSettingsButton.addActionListener(this::handleSaveSettings);
        
        connectButton = new JButton("Connect All");
        connectButton.addActionListener(this::handleConnect);
        
        buttonPanel.add(saveSettingsButton);
        buttonPanel.add(connectButton);
        
        panel.add(new JScrollPane(configGrid), BorderLayout.CENTER);
        panel.add(buttonPanel, BorderLayout.SOUTH);
        
        return panel;
    }
    
    private JPanel createQueryPanel() {
        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        // Query input area
        JPanel queryPanel = new JPanel(new BorderLayout());
        queryPanel.setBorder(BorderFactory.createTitledBorder("Ask a Question (RAG-Enhanced)"));
        
        queryTextArea = new JTextArea(4, 40);
        queryTextArea.setLineWrap(true);
        queryTextArea.setWrapStyleWord(true);
        queryTextArea.setText("What are all the classes in my ontology and how are they related?");
        queryPanel.add(new JScrollPane(queryTextArea), BorderLayout.CENTER);
        
        executeButton = new JButton("Execute RAG Query");
        executeButton.addActionListener(this::handleExecuteQuery);
        queryPanel.add(executeButton, BorderLayout.SOUTH);
        
        // Results area
        JPanel resultsPanel = new JPanel(new BorderLayout());
        resultsPanel.setBorder(BorderFactory.createTitledBorder("Results & Explanation"));
        
        resultsTextArea = new JTextArea(20, 40);
        resultsTextArea.setEditable(false);
        resultsTextArea.setLineWrap(true);
        resultsTextArea.setWrapStyleWord(true);
        resultsPanel.add(new JScrollPane(resultsTextArea), BorderLayout.CENTER);
        
        JSplitPane splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, queryPanel, resultsPanel);
        splitPane.setDividerLocation(150);
        
        panel.add(splitPane, BorderLayout.CENTER);
        
        return panel;
    }
    
    private JPanel createVectorStorePanel() {
        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        // Top panel with chunking strategy and buttons
        JPanel topPanel = new JPanel(new BorderLayout());
        
        // Chunking strategy selection
        JPanel chunkingPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        chunkingPanel.add(new JLabel("Chunking Strategy:"));
        String[] chunkingStrategies = {
            // Text-based (for Neo4j text data)
            "WordChunking",
            "SentenceChunking",
            "ParagraphChunking",
            "FixedSizeChunking",
            // OWL-aware (for ontology structures)
            "ClassBasedChunking",
            "AnnotationBasedChunking",
            "NamespaceBasedChunking",
            "DepthBasedChunking",
            "ModuleExtractionChunking",
            "SizeBasedChunking"
        };
        chunkingStrategyCombo = new JComboBox<>(chunkingStrategies);
        chunkingStrategyCombo.setToolTipText("Text chunking (first 4) for Neo4j; OWL chunking (last 6) for ontologies");
        chunkingPanel.add(chunkingStrategyCombo);
        
        topPanel.add(chunkingPanel, BorderLayout.NORTH);
        
        // Control buttons
        JPanel controlPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        
        JButton indexGraphButton = new JButton("Index Neo4j Graph to Vector Store");
        indexGraphButton.addActionListener(this::handleIndexGraph);
        indexGraphButton.setToolTipText("Convert Neo4j graph to embeddings and store in Lucene index");
        
        JButton indexOntologyButton = new JButton("Index Protégé Ontology to Vector Store");
        indexOntologyButton.addActionListener(this::handleIndexOntology);
        indexOntologyButton.setToolTipText("Convert loaded OWL ontology to embeddings and store in Lucene index");
        
        JButton viewStatsButton = new JButton("View Vector Store Stats");
        viewStatsButton.addActionListener(this::handleViewStats);
        
        JButton clearIndexButton = new JButton("Clear Index");
        clearIndexButton.addActionListener(this::handleClearIndex);
        clearIndexButton.setToolTipText("Delete all vectors from Lucene index");
        
        controlPanel.add(indexGraphButton);
        controlPanel.add(indexOntologyButton);
        controlPanel.add(viewStatsButton);
        controlPanel.add(clearIndexButton);
        
        topPanel.add(controlPanel, BorderLayout.CENTER);
        
        statsArea = new JTextArea(20, 40);
        statsArea.setEditable(false);
        statsArea.setText("Lucene Vector Store Statistics:\n\nClick 'View Vector Store Stats' to see current status.");
        
        panel.add(topPanel, BorderLayout.NORTH);
        panel.add(new JScrollPane(statsArea), BorderLayout.CENTER);
        
        return panel;
    }
    
    private void addSectionHeader(JPanel panel, GridBagConstraints gbc, int row, String title) {
        gbc.gridy = row;
        gbc.gridx = 0;
        gbc.gridwidth = 2;
        JLabel header = new JLabel(title);
        header.setFont(header.getFont().deriveFont(Font.BOLD, 14f));
        panel.add(header, gbc);
        gbc.gridwidth = 1;
    }
    
    private void addLabeledField(JPanel panel, GridBagConstraints gbc, String label, JComponent field) {
        gbc.gridx = 0;
        gbc.weightx = 0.0;
        panel.add(new JLabel(label), gbc);
        
        gbc.gridx = 1;
        gbc.weightx = 1.0;
        panel.add(field, gbc);
    }
    
    private void handleSaveSettings(ActionEvent e) {
        // Save Neo4j settings
        preferences.setNeo4jUri(neo4jUriField.getText());
        preferences.setNeo4jUsername(neo4jUsernameField.getText());
        preferences.setNeo4jPassword(new String(neo4jPasswordField.getPassword()));
        preferences.setNeo4jDatabase(neo4jDatabaseField.getText());
        
        // Save Lucene settings
        preferences.setLuceneIndexPath(luceneIndexPathField.getText());
        
        // Save model settings
        preferences.setEmbeddingModel((String) embeddingModelCombo.getSelectedItem());
        preferences.setEmbeddingApiKey(new String(embeddingApiKeyField.getPassword()));
        preferences.setAiModel((String) aiModelCombo.getSelectedItem());
        preferences.setAiApiKey(new String(aiApiKeyField.getPassword()));
        preferences.setChunkingStrategy((String) chunkingStrategyCombo.getSelectedItem());
        
        JOptionPane.showMessageDialog(this, "Settings saved successfully!", 
            "Success", JOptionPane.INFORMATION_MESSAGE);
        
        logger.info("Settings saved to preferences");
    }
    
    private void handleConnect(ActionEvent e) {
        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() throws Exception {
                // Initialize RAG service with all components
                ragService = new RagService(
                    neo4jUriField.getText(),
                    neo4jUsernameField.getText(),
                    new String(neo4jPasswordField.getPassword()),
                    neo4jDatabaseField.getText(),
                    luceneIndexPathField.getText(),
                    (String) embeddingModelCombo.getSelectedItem(),
                    new String(embeddingApiKeyField.getPassword()),
                    (String) aiModelCombo.getSelectedItem(),
                    new String(aiApiKeyField.getPassword()),
                    (String) chunkingStrategyCombo.getSelectedItem()
                );
                
                return null;
            }
            
            @Override
            protected void done() {
                try {
                    get();
                    // Set the current ontology in RagService
                    if (getOWLModelManager() != null && getOWLModelManager().getActiveOntology() != null) {
                        ragService.setOntology(getOWLModelManager().getActiveOntology());
                    }
                    neo4jStatusLabel.setText("🟢 Connected");
                    vectorStoreStatusLabel.setText("🟢 Ready (File-based)");
                    JOptionPane.showMessageDialog(RagQueryPanel.this,
                        "Successfully connected to all services!\nLucene index: " + luceneIndexPathField.getText(),
                        "Connected", JOptionPane.INFORMATION_MESSAGE);
                    
                    // Auto-refresh stats
                    handleViewStats(null);
                } catch (Exception ex) {
                    neo4jStatusLabel.setText("🔴 Connection Failed");
                    vectorStoreStatusLabel.setText("🔴 Failed");
                    JOptionPane.showMessageDialog(RagQueryPanel.this,
                        "Connection failed: " + ex.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
                    logger.error("Connection failed", ex);
                }
            }
        }.execute();
    }
    
    private void handleExecuteQuery(ActionEvent e) {
        if (ragService == null) {
            JOptionPane.showMessageDialog(this,
                "Please connect to services first!",
                "Not Connected", JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        String query = queryTextArea.getText().trim();
        if (query.isEmpty()) {
            return;
        }
        
        resultsTextArea.setText("Processing RAG query...\n\n");
        executeButton.setEnabled(false);
        
        new SwingWorker<String, Void>() {
            @Override
            protected String doInBackground() throws Exception {
                return ragService.executeRagQuery(query, 5);
            }
            
            @Override
            protected void done() {
                try {
                    String result = get();
                    resultsTextArea.setText(result);
                } catch (Exception ex) {
                    resultsTextArea.setText("Error: " + ex.getMessage());
                    logger.error("RAG query failed", ex);
                } finally {
                    executeButton.setEnabled(true);
                }
            }
        }.execute();
    }
    
    private void handleIndexGraph(ActionEvent e) {
        if (ragService == null) {
            JOptionPane.showMessageDialog(this,
                "Please connect to services first!",
                "Not Connected", JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        // Update chunking strategy from UI
        String selectedStrategy = (String) chunkingStrategyCombo.getSelectedItem();
        ragService.setChunkingStrategy(selectedStrategy);
        
        statsArea.setText("Indexing Neo4j graph with " + selectedStrategy + "...\n\nThis may take a few moments...");
        
        new SwingWorker<Integer, Void>() {
            @Override
            protected Integer doInBackground() throws Exception {
                logger.info("Starting Neo4j graph indexing...");
                return ragService.indexGraphToVectorStore();
            }
            
            @Override
            protected void done() {
                try {
                    int count = get();
                    JOptionPane.showMessageDialog(RagQueryPanel.this,
                        "Successfully indexed Neo4j graph!\n\nIndexed " + count + " chunks",
                        "Indexing Complete", JOptionPane.INFORMATION_MESSAGE);
                    logger.info("Graph indexing completed successfully: {} chunks", count);
                    
                    // Auto-refresh stats
                    handleViewStats(null);
                } catch (Exception ex) {
                    JOptionPane.showMessageDialog(RagQueryPanel.this,
                        "Graph indexing failed: " + ex.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
                    logger.error("Graph indexing failed", ex);
                    statsArea.setText("Indexing failed: " + ex.getMessage());
                }
            }
        }.execute();
    }
    
    private void handleIndexOntology(ActionEvent e) {
        if (ragService == null) {
            JOptionPane.showMessageDialog(this,
                "Please connect to services first!",
                "Not Connected", JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        // Update ontology in case it changed
        if (getOWLModelManager() == null || getOWLModelManager().getActiveOntology() == null) {
            JOptionPane.showMessageDialog(this,
                "No ontology loaded in Protégé!",
                "No Ontology", JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        // Update chunking strategy from UI
        String selectedStrategy = (String) chunkingStrategyCombo.getSelectedItem();
        ragService.setChunkingStrategy(selectedStrategy);
        
        ragService.setOntology(getOWLModelManager().getActiveOntology());
        statsArea.setText("Indexing Protégé ontology with " + selectedStrategy + "...\n\nThis may take a few moments...");
        
        new SwingWorker<Integer, Void>() {
            @Override
            protected Integer doInBackground() throws Exception {
                logger.info("Starting Protégé ontology indexing...");
                return ragService.indexOntologyToVectorStore();
            }
            
            @Override
            protected void done() {
                try {
                    int count = get();
                    JOptionPane.showMessageDialog(RagQueryPanel.this,
                        "Successfully indexed Protégé ontology!\n\nIndexed " + count + " chunks",
                        "Indexing Complete", JOptionPane.INFORMATION_MESSAGE);
                    logger.info("Ontology indexing completed successfully: {} chunks", count);
                    
                    // Auto-refresh stats
                    handleViewStats(null);
                } catch (Exception ex) {
                    JOptionPane.showMessageDialog(RagQueryPanel.this,
                        "Ontology indexing failed: " + ex.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
                    logger.error("Ontology indexing failed", ex);
                    statsArea.setText("Indexing failed: " + ex.getMessage());
                }
            }
        }.execute();
    }
    
    private void handleViewStats(ActionEvent e) {
        if (ragService == null) {
            statsArea.setText("Lucene Vector Store Statistics:\n\nStatus: Not initialized\n\nPlease connect to services first.");
            return;
        }
        
        try {
            LuceneVectorStore.IndexStats stats = ragService.getVectorStoreStats();
            StringBuilder sb = new StringBuilder();
            sb.append("Lucene Vector Store Statistics:\n\n");
            sb.append("Index Path: ").append(stats.getIndexPath()).append("\n");
            sb.append("Total Documents: ").append(stats.getDocumentCount()).append("\n");
            sb.append("Vector Dimension: ").append(stats.getVectorDimension()).append("\n");
            sb.append("Storage Type: File-based (Lucene)\n");
            sb.append("Search Algorithm: HNSW (Cosine Similarity)\n");
            sb.append("Status: ").append(stats.getDocumentCount() > 0 ? "Ready" : "Empty - Please index data").append("\n\n");
            
            if (stats.getDocumentCount() == 0) {
                sb.append("No vectors indexed yet.\n");
                sb.append("Use 'Index Neo4j Graph' button above to populate the index.");
            } else {
                sb.append("Vector store is ready for queries.\n");
                sb.append("All data persisted to disk.");
            }
            
            statsArea.setText(sb.toString());
            logger.info("Displayed vector store stats: {} documents", stats.getDocumentCount());
        } catch (Exception ex) {
            statsArea.setText("Error retrieving stats: " + ex.getMessage());
            logger.error("Failed to get vector store stats", ex);
        }
    }
    
    private void handleClearIndex(ActionEvent e) {
        if (ragService == null) {
            JOptionPane.showMessageDialog(this,
                "Please connect to services first!",
                "Not Connected", JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        int confirm = JOptionPane.showConfirmDialog(this,
            "Are you sure you want to clear the entire index?\nThis cannot be undone.",
            "Confirm Clear Index",
            JOptionPane.YES_NO_OPTION,
            JOptionPane.WARNING_MESSAGE);
        
        if (confirm == JOptionPane.YES_OPTION) {
            try {
                ragService.clearVectorStore();
                statsArea.setText("Index cleared successfully.\n\nThe index is now empty.");
                JOptionPane.showMessageDialog(this,
                    "Index cleared successfully!",
                    "Success", JOptionPane.INFORMATION_MESSAGE);
                logger.info("Index cleared by user");
                
                // Refresh stats
                handleViewStats(null);
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this,
                    "Failed to clear index: " + ex.getMessage(),
                    "Error", JOptionPane.ERROR_MESSAGE);
                logger.error("Failed to clear index", ex);
            }
        }
    }
    
    private void loadPreferences() {
        neo4jUriField.setText(preferences.getNeo4jUri());
        neo4jUsernameField.setText(preferences.getNeo4jUsername());
        neo4jPasswordField.setText(preferences.getNeo4jPassword());
        neo4jDatabaseField.setText(preferences.getNeo4jDatabase());
        
        luceneIndexPathField.setText(preferences.getLuceneIndexPath());
        
        String embModel = preferences.getEmbeddingModel();
        if (embModel != null) {
            embeddingModelCombo.setSelectedItem(embModel);
        }
        embeddingApiKeyField.setText(preferences.getEmbeddingApiKey());
        
        String aiModel = preferences.getAiModel();
        if (aiModel != null) {
            aiModelCombo.setSelectedItem(aiModel);
        }
        aiApiKeyField.setText(preferences.getAiApiKey());
        
        String chunkingStrategy = preferences.getChunkingStrategy();
        if (chunkingStrategy != null) {
            chunkingStrategyCombo.setSelectedItem(chunkingStrategy);
        }
    }
    
    @Override
    protected void disposeOWLView() {
        if (ragService != null) {
            ragService.close();
        }
    }
}
