"""
FertilityGraphRAG - Version LOCAL (Windows/Mac/Linux)
Adapté depuis la version Colab
"""
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import zipfile
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import re

import networkx as nx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class FertilityGraphRAG:
    """
    GraphRAG Complet pour analyse de fertilité
    Version locale - pas de dépendances Google Colab
    """

    def __init__(self, pdf_directory: str = "./fertility_docs"):
        """
        Initialise GraphRAG
        
        Args:
            pdf_directory: chemin vers le dossier contenant les PDFs
        """
        self.pdf_dir = Path(pdf_directory)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Configurations
        self.chunk_size = 800
        self.chunk_overlap = 100
        self.top_k_retrieval = 5

        # Stockage
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.vector_store = None
        self.graph = nx.DiGraph()

        # Entités médicales spécialisées fertilité
        self.medical_entities = {
            'hormones': [
                'AMH', 'FSH', 'LH', 'estradiol', 'progesterone',
                'testosterone', 'prolactin', 'thyroid', 'TSH'
            ],
            'conditions': [
                'PCOS', 'endometriosis', 'ovarian reserve', 'infertility',
                'anovulation', 'oligomenorrhea', 'amenorrhea', 'POI',
                'premature ovarian insufficiency', 'diminished ovarian reserve'
            ],
            'measurements': [
                'ng/mL', 'mIU/mL', 'pmol/L', 'IU/L', 'ng/dL',
                'follicle count', 'antral follicle', 'AFC'
            ],
            'age_groups': [
                'under 35', '35-40', 'over 40', 'advanced maternal age'
            ],
            'treatments': [
                'IVF', 'IUI', 'ovulation induction', 'letrozole',
                'clomid', 'gonadotropins', 'metformin', 'egg freezing'
            ],
            'tests': [
                'ultrasound', 'HSG', 'semen analysis', 'hormone panel',
                'ovarian reserve testing', 'AFC count'
            ]
        }

        print("📚 FertilityGraphRAG initialisé")
        print(f"📁 Répertoire PDFs: {self.pdf_dir.absolute()}")

    # ========================================================================
    # ÉTAPE 1: CHARGEMENT DES PDFs
    # ========================================================================

    def step1_load_pdfs(self) -> List:
        """Charge tous les PDFs du répertoire"""
        print("\n" + "="*70)
        print("ÉTAPE 1: CHARGEMENT DES PDFs")
        print("="*70)

        pdf_files = list(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"⚠️  Aucun PDF trouvé dans {self.pdf_dir}")
            print("\n📝 Instructions:")
            print(f"   1. Créez le dossier: {self.pdf_dir}")
            print("   2. Placez vos PDFs médicaux dedans")
            print("   3. Relancez le script")
            return []

        print(f"📄 {len(pdf_files)} PDF(s) trouvé(s)")

        all_documents = []
        for pdf_path in pdf_files:
            try:
                print(f"\n   📖 Chargement: {pdf_path.name}")
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()

                # Ajouter métadonnées
                for doc in docs:
                    doc.metadata['filename'] = pdf_path.name
                    doc.metadata['file_path'] = str(pdf_path)

                all_documents.extend(docs)
                print(f"      ✓ {len(docs)} pages chargées")

            except Exception as e:
                print(f"      ✗ Erreur: {e}")

        self.documents = all_documents
        print(f"\n✅ Total: {len(all_documents)} pages chargées")
        return all_documents

    # ========================================================================
    # ÉTAPE 2: CHUNKING
    # ========================================================================

    def step2_create_chunks(self) -> List:
        """Découpe les documents en chunks"""
        print("\n" + "="*70)
        print("ÉTAPE 2: CHUNKING INTELLIGENT")
        print("="*70)

        if not self.documents:
            print("⚠️  Aucun document chargé")
            return []

        print(f"⚙️  Configuration:")
        print(f"   - Taille chunk: {self.chunk_size} caractères")
        print(f"   - Overlap: {self.chunk_overlap} caractères")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            length_function=len,
        )

        print(f"\n🔄 Découpage en cours...")
        chunks = text_splitter.split_documents(self.documents)

        # Enrichir métadonnées
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = f"chunk_{i}"
            chunk.metadata['chunk_length'] = len(chunk.page_content)

        self.chunks = chunks

        print(f"✅ {len(chunks)} chunks créés")
        print(f"\n📊 Statistiques:")
        print(f"   - Taille moyenne: {sum(len(c.page_content) for c in chunks) // len(chunks)} caractères")
        print(f"   - Plus petit: {min(len(c.page_content) for c in chunks)} caractères")
        print(f"   - Plus grand: {max(len(c.page_content) for c in chunks)} caractères")

        if chunks:
            print(f"\n💡 Exemple de chunk:")
            print(f"   {chunks[0].page_content[:200]}...")

        return chunks

    # ========================================================================
    # ÉTAPE 3: EMBEDDINGS
    # ========================================================================

    def step3_create_embeddings(self):
        """Crée le modèle d'embeddings"""
        print("\n" + "="*70)
        print("ÉTAPE 3: CRÉATION DES EMBEDDINGS")
        print("="*70)

        print("🔄 Chargement du modèle d'embeddings...")
        print("   Modèle: sentence-transformers/all-MiniLM-L6-v2")
        print("   - Taille: ~90MB")
        print("   - Dimension: 384")
        print("   - Optimisé pour CPU")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        print("✅ Modèle d'embeddings chargé")

        # Test
        test_text = "AMH level of 1.1 ng/mL in PCOS patient"
        test_embedding = self.embeddings.embed_query(test_text)
        print(f"\n🧪 Test d'embedding:")
        print(f"   Texte: '{test_text}'")
        print(f"   Dimension: {len(test_embedding)}")

        return self.embeddings

    # ========================================================================
    # ÉTAPE 4: KNOWLEDGE GRAPH
    # ========================================================================

    def step4_build_knowledge_graph(self):
        """Construit le knowledge graph"""
        print("\n" + "="*70)
        print("ÉTAPE 4: CONSTRUCTION DU KNOWLEDGE GRAPH")
        print("="*70)

        if not self.chunks:
            print("⚠️  Aucun chunk disponible")
            return

        print("🔄 Construction du graphe...")

        entity_counts = defaultdict(int)

        for i, chunk in enumerate(self.chunks):
            chunk_id = chunk.metadata['chunk_id']
            text = chunk.page_content.lower()

            # Ajouter nœud chunk
            self.graph.add_node(
                chunk_id,
                type='chunk',
                text=chunk.page_content[:300],
                full_text=chunk.page_content,
                source=chunk.metadata.get('filename', 'unknown'),
                page=chunk.metadata.get('page', 0),
                chunk_length=len(chunk.page_content)
            )

            # Extraire entités
            chunk_entities = self.extract_entities_advanced(text)

            for category, entity_list in chunk_entities.items():
                for entity in entity_list:
                    entity_node = f"entity_{category}_{entity.lower().replace(' ', '_')}"

                    entity_counts[entity] += 1

                    if not self.graph.has_node(entity_node):
                        self.graph.add_node(
                            entity_node,
                            type='entity',
                            category=category,
                            name=entity,
                            occurrences=1
                        )
                    else:
                        self.graph.nodes[entity_node]['occurrences'] += 1

                    self.graph.add_edge(
                        chunk_id,
                        entity_node,
                        relation='mentions',
                        weight=1.0
                    )

            if (i + 1) % 50 == 0:
                print(f"   📈 Traité: {i + 1}/{len(self.chunks)} chunks")

        # Relations entre entités
        self._create_entity_cooccurrence_edges()

        print(f"\n✅ Knowledge Graph construit")
        print(f"\n📊 Statistiques:")
        print(f"   - Nœuds totaux: {self.graph.number_of_nodes()}")
        print(f"   - Arêtes totales: {self.graph.number_of_edges()}")
        print(f"   - Nœuds 'chunk': {sum(1 for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'chunk')}")
        print(f"   - Nœuds 'entity': {sum(1 for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'entity')}")

        print(f"\n🏆 Top 5 entités:")
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for entity, count in top_entities:
            print(f"   - {entity}: {count} fois")

        return self.graph

    def extract_entities_advanced(self, text: str) -> Dict[str, List[str]]:
        """Extraction d'entités avec regex"""
        found_entities = defaultdict(list)

        for category, entities in self.medical_entities.items():
            for entity in entities:
                pattern = r'\b' + re.escape(entity.lower()) + r'\b'
                if re.search(pattern, text):
                    found_entities[category].append(entity)

        # Valeurs numériques
        numeric_patterns = [
            (r'(\d+\.?\d*)\s*(ng/ml|ng/dl|miu/ml|iu/l|pmol/l)', 'measurements'),
            (r'(\d+)\s*follicles?', 'measurements'),
        ]

        for pattern, category in numeric_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(0)
                if value not in found_entities[category]:
                    found_entities[category].append(value)

        return dict(found_entities)

    def _create_entity_cooccurrence_edges(self):
        """Crée relations entre entités co-occurrentes"""
        print("\n   🔗 Création des relations entre entités...")

        entity_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'entity']

        cooccurrence_count = 0
        for entity1 in entity_nodes:
            chunks_with_entity1 = list(self.graph.predecessors(entity1))

            for entity2 in entity_nodes:
                if entity1 >= entity2:
                    continue

                chunks_with_entity2 = list(self.graph.predecessors(entity2))
                common_chunks = set(chunks_with_entity1) & set(chunks_with_entity2)

                if common_chunks:
                    weight = len(common_chunks) / max(len(chunks_with_entity1), len(chunks_with_entity2))
                    self.graph.add_edge(
                        entity1,
                        entity2,
                        relation='co_occurs_with',
                        weight=weight,
                        common_chunks=len(common_chunks)
                    )
                    cooccurrence_count += 1

        print(f"      ✓ {cooccurrence_count} relations créées")

    # ========================================================================
    # ÉTAPE 5: VECTOR STORE
    # ========================================================================

    def step5_build_vector_store(self):
        """Construit le vector store"""
        print("\n" + "="*70)
        print("ÉTAPE 5: CONSTRUCTION DU VECTOR STORE")
        print("="*70)

        if not self.chunks:
            print("⚠️  Aucun chunk disponible")
            return

        if not self.embeddings:
            print("⚠️  Embeddings non créés")
            return

        print("🔄 Vectorisation des chunks...")
        print(f"   - {len(self.chunks)} chunks à vectoriser")
        print("   - Cela peut prendre 1-2 minutes...")

        # Supprimer ancienne DB
        chroma_dir = Path("./chroma_db")
        if chroma_dir.exists():
            import shutil
            shutil.rmtree(chroma_dir)
            print("   🗑️  Ancienne base supprimée")

        # Créer vector store
        self.vector_store = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db",
            collection_name="fertility_collection"
        )

        print(f"✅ Vector store créé")
        print(f"   - Collection: fertility_collection")
        print(f"   - Répertoire: ./chroma_db")
        print(f"   - {len(self.chunks)} vecteurs stockés")

        # Test
        print("\n🧪 Test de recherche:")
        test_query = "What is a normal AMH level?"
        results = self.vector_store.similarity_search(test_query, k=2)
        print(f"   Query: '{test_query}'")
        print(f"   Résultats: {len(results)}")
        if results:
            print(f"   Premier: {results[0].page_content[:100]}...")

        return self.vector_store

    # ========================================================================
    # ÉTAPE 6: RETRIEVAL HYBRIDE
    # ========================================================================

    def step6_retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Récupération hybride (Vector + Graph)"""
        print(f"\n🔍 Query: '{query}'")

        if not self.vector_store:
            return {"error": "Vector store non initialisé"}

        # 1. VECTOR SEARCH
        print("\n   1️⃣ Vector Search...")
        vector_results = self.vector_store.similarity_search_with_score(query, k=top_k)
        print(f"      ✓ {len(vector_results)} résultats")

        # 2. GRAPH SEARCH (seulement si le graph existe)
        graph_chunks = []
        entity_info = []
        query_entities = {}
        
        if self.graph.number_of_nodes() > 0:
            print("   2️⃣ Graph Search...")
            query_entities = self.extract_entities_advanced(query.lower())

            for category, entities in query_entities.items():
                for entity in entities:
                    entity_node = f"entity_{category}_{entity.lower().replace(' ', '_')}"

                    if self.graph.has_node(entity_node):
                        node_data = self.graph.nodes[entity_node]
                        entity_info.append({
                            'entity': entity,
                            'category': category,
                            'occurrences': node_data.get('occurrences', 0)
                        })

                        connected_chunks = list(self.graph.predecessors(entity_node))

                        for chunk_id in connected_chunks[:3]:
                            chunk_text = self.graph.nodes[chunk_id].get('full_text', '')
                            if chunk_text:
                                graph_chunks.append({
                                    'chunk_id': chunk_id,
                                    'text': chunk_text,
                                    'entity': entity,
                                    'source': self.graph.nodes[chunk_id].get('source', 'unknown')
                                })

            print(f"      ✓ {len(graph_chunks)} chunks via graphe")
        else:
            print("   2️⃣ Graph Search: SKIPPED (cache mode - vector search only)")

        return {
            'query': query,
            'vector_results': [
                {
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score)
                }
                for doc, score in vector_results
            ],
            'graph_results': graph_chunks[:5],
            'detected_entities': query_entities,
            'entity_info': entity_info
        }

    def format_context_for_llm(self, retrieval_results: Dict[str, Any]) -> str:
        """Formate le contexte pour le LLM"""
        context_parts = []

        context_parts.append("=== MEDICAL CONTEXT ===\n")
        context_parts.append(f"Question: {retrieval_results['query']}\n")

        # Entités
        if retrieval_results['detected_entities']:
            context_parts.append("\n--- DETECTED ENTITIES ---")
            for category, entities in retrieval_results['detected_entities'].items():
                context_parts.append(f"{category.upper()}: {', '.join(entities)}")

        # Vector results
        context_parts.append("\n--- RELEVANT INFORMATION (Vector Search) ---")
        for i, result in enumerate(retrieval_results['vector_results'][:3], 1):
            source = result['metadata'].get('filename', 'unknown')
            page = result['metadata'].get('page', '?')
            score = result['score']
            context_parts.append(
                f"\n[{i}] Source: {source} (page {page}) | Score: {score:.3f}\n"
                f"{result['text']}"
            )

        # Graph results
        if retrieval_results['graph_results']:
            context_parts.append("\n--- RELATED INFORMATION (Graph Search) ---")
            for i, result in enumerate(retrieval_results['graph_results'][:2], 1):
                context_parts.append(
                    f"\n[{i}] Entity: {result['entity']} | Source: {result['source']}\n"
                    f"{result['text'][:400]}..."
                )

        context_parts.append("\n=== END CONTEXT ===")

        return "\n".join(context_parts)

    # ========================================================================
    # PIPELINE COMPLET
    # ========================================================================

    def run_full_pipeline(self):
        """Exécute le pipeline complet"""
        print("\n" + "🚀"*35)
        print("PIPELINE GRAPHRAG COMPLET")
        print("🚀"*35)

        docs = self.step1_load_pdfs()
        if not docs:
            return False

        self.step2_create_chunks()
        self.step3_create_embeddings()
        self.step4_build_knowledge_graph()
        self.step5_build_vector_store()

        print("\n" + "="*70)
        print("✅ PIPELINE TERMINÉ")
        print("="*70)

        return True


# ========================================================================
# MAIN - LANCEMENT DU SCRIPT
# ========================================================================

if __name__ == "__main__":
    print("🚀 Démarrage de FertilityGraphRAG...\n")
    
    # Créer l'instance
    graphrag = FertilityGraphRAG(pdf_directory="./fertility_docs")
    
    # Lancer le pipeline complet
    success = graphrag.run_full_pipeline()
    
    if success:
        print("\n" + "="*70)
        print("✅ SYSTÈME PRÊT - TEST DE RECHERCHE")
        print("="*70)
        
        # Test
        test_query = "What is AMH?"
        print(f"\n🔍 Test Query: '{test_query}'")
        results = graphrag.step6_retrieve(test_query, top_k=3)
        
        print(f"\n📊 Résultats:")
        print(f"   - Vector results: {len(results.get('vector_results', []))}")
        print(f"   - Graph results: {len(results.get('graph_results', []))}")
        print(f"   - Entités détectées: {results.get('detected_entities', {})}")
        
        # Afficher le contexte formaté
        print("\n📄 Contexte formaté pour LLM:")
        context = graphrag.format_context_for_llm(results)
        print(context[:800] + "...\n")
    else:
        print("\n❌ ERREUR: Aucun PDF trouvé")
        print("📝 Placez vos PDFs médicaux dans ./fertility_docs")
