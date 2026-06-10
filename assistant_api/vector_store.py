"""
Модуль работы с векторным хранилищем ChromaDB.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from pypdf import PdfReader

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Инициализация ChromaDB клиента
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Создана новая коллекция '{collection_name}'")
        
        # OpenAI клиент для создания embeddings
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Пост-обработка: фильтруем слишком короткие чанки
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _read_text_file(self, file_path: str) -> str:
        """
        Чтение обычного текстового файла.
        Подходит для .txt, .md, .csv, .html.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _read_pdf_file(self, file_path: str) -> str:
        """
        Извлечение текста из PDF-файла.
        """
        reader = PdfReader(file_path)
        pages_text = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        return "\n\n".join(pages_text)

    def _read_document(self, file_path: str) -> str:
        """
        Чтение документа в зависимости от расширения файла.
        """
        extension = Path(file_path).suffix.lower()

        if extension in [".txt", ".md", ".csv", ".html"]:
            return self._read_text_file(file_path)

        if extension == ".pdf":
            return self._read_pdf_file(file_path)

        raise ValueError(f"Неподдерживаемый формат файла: {extension}")

    def _get_supported_files(self, path: str) -> List[str]:
        """
        Получение списка поддерживаемых файлов.
        Можно передать путь к одному файлу или к папке.
        """
        supported_extensions = {".txt", ".md", ".csv", ".html", ".pdf"}

        if os.path.isfile(path):
            return [path] if Path(path).suffix.lower() in supported_extensions else []

        if os.path.isdir(path):
            files = []
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name)
                if os.path.isfile(file_path) and Path(file_path).suffix.lower() in supported_extensions:
                    files.append(file_path)

            return sorted(files)

        return []

    def load_documents(self, file_path: str):
        """
        Загрузка документов из файла или папки в векторное хранилище.

        Поддерживаются форматы:
        - .txt
        - .md
        - .csv
        - .html
        - .pdf

        Args:
            file_path: путь к файлу или папке с документами
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Путь {file_path} не найден")

        # Проверка, не загружены ли уже документы
        if self.collection.count() > 0:
            print("Документы уже загружены в коллекцию")
            return

        files = self._get_supported_files(file_path)

        if not files:
            raise ValueError(
                f"В папке или файле {file_path} не найдено поддерживаемых документов"
            )

        print(f"Найдено файлов для загрузки: {len(files)}")

        documents = []
        ids = []
        embeddings = []
        metadatas = []

        for file_index, current_file_path in enumerate(files):
            print(f"\nЧтение файла: {current_file_path}")

            text = self._read_document(current_file_path)

            if not text.strip():
                print(f"Файл пропущен, текст не найден: {current_file_path}")
                continue

            chunks = self._chunk_text(text)
            print(f"Файл разбит на {len(chunks)} чанков")

            source_name = os.path.basename(current_file_path)
            file_type = Path(current_file_path).suffix.lower()

            for chunk_index, chunk in enumerate(chunks):
                embedding = self._create_embedding(chunk)

                documents.append(chunk)
                ids.append(f"doc_{file_index}_{chunk_index}")
                embeddings.append(embedding)
                metadatas.append({
                    "source": source_name,
                    "file_type": file_type
                })

                if len(documents) % 10 == 0:
                    print(f"Обработано {len(documents)} чанков")

        if not documents:
            raise ValueError("Не удалось извлечь текст ни из одного документа")

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        print(f"\nЗагружено {len(documents)} чанков в коллекцию '{self.collection_name}'")
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста через OpenAI.
        
        Args:
            text: текст для векторизации
            
        Returns:
            вектор embeddings
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.

        Args:
            query: текст запроса
            top_k: количество документов для возврата

        Returns:
            список документов с текстом, метаданными и расстоянием
        """
        query_embedding = self._create_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        documents = []

        if results["documents"] and len(results["documents"]) > 0:
            for i in range(len(results["documents"][0])):
                metadata = None

                if results.get("metadatas") and results["metadatas"][0]:
                    metadata = results["metadatas"][0][i]

                documents.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": metadata,
                    "distance": results["distances"][0][i] if "distances" in results else None
                })

        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")
    
    # Загрузка документов
    if os.path.exists("data/docs.txt"):
        vector_store.load_documents("data/docs.txt")
    
    # Поиск
    results = vector_store.search("Что такое машинное обучение?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")

