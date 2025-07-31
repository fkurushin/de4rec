import sys
sys.path.append("src")

import pytest
import time
import psutil
import os
import gc
from de4rec import DualEncoderDatasets

class DualEncoderLoadData(DualEncoderDatasets):
    def __init__(self, sparse_neg_sampling=True, **kwargs):
        _interactions_path = kwargs.get("interactions_path", "dataset/ml-1m/ratings.dat")
        assert _interactions_path
        _interactions = self.load_list_of_int_int_from_path(_interactions_path)
        
        _users_size = max([tu[0] for tu in _interactions]) + 1
        _items_size = max([tu[1] for tu in _interactions]) + 1
        
        super().__init__(
            interactions=_interactions, 
            users_size=_users_size, 
            items_size=_items_size, 
            freq_margin=0.1, 
            neg_per_sample=1, 
            sparse_neg_sampling=sparse_neg_sampling
        )
    
    def load_list_of_int_int_from_path(self, path: str, sep: str = "::") -> list[tuple[int, int]]:
        with open(path, "r", encoding="utf-8") as fn:
            res = list(
                map(
                    lambda row: (int(row[0]), int(row[1])),
                    map(
                        lambda row: row.strip().split(sep)[:2],
                        fn.read().strip().split("\n"),
                    ),
                )
            )
        return res

class TestSamplingBenchmarks:
    def get_memory_usage(self):
        """Получить текущее использование памяти в MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_sparse_vs_dense_sampling_speed(self):
        """Сравнение скорости sparse vs dense негативного семплирования"""
        
        # Тест 1: Sparse sampling
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        datasets_sparse = DualEncoderLoadData(
            interactions_path="dataset/ml-1m/ratings.dat",
            sparse_neg_sampling=True
        )
        
        sparse_time = time.time() - start_time
        sparse_memory = self.get_memory_usage() - start_memory
        
        # Тест 2: Dense sampling  
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        datasets_dense = DualEncoderLoadData(
            interactions_path="dataset/ml-1m/ratings.dat",
            sparse_neg_sampling=False
        )
        
        dense_time = time.time() - start_time
        dense_memory = self.get_memory_usage() - start_memory
        
        # Вывод результатов
        print(f"\n=== Sampling Speed Comparison ===")
        print(f"Sparse sampling:  {sparse_time:.4f}s")
        print(f"Dense sampling:   {dense_time:.4f}s")
        print(f"Speed ratio:      {dense_time/sparse_time:.2f}x")
        
        print(f"\n=== Memory Usage Comparison ===")
        print(f"Sparse sampling:  {sparse_memory:.2f} MB")
        print(f"Dense sampling:   {dense_memory:.2f} MB")
        print(f"Memory ratio:     {dense_memory/sparse_memory:.2f}x")
        
        # Проверяем, что результаты одинаковые
        sparse_train_size = len(datasets_sparse.dataset_split.train_dataset)
        dense_train_size = len(datasets_dense.dataset_split.train_dataset)
        
        print(f"\n=== Dataset Size Comparison ===")
        print(f"Sparse train size: {sparse_train_size}")
        print(f"Dense train size:  {dense_train_size}")
        
        # Assertions
        assert sparse_train_size > 0
        assert dense_train_size > 0
        # Sparse должен быть быстрее или не сильно медленнее
        assert sparse_time < dense_time * 1.5  # Sparse не более чем в 1.5 раза медленнее
    
    def test_sparse_benchmark(self, benchmark):
        """Бенчмарк sparse негативного семплирования"""
        def create_sparse_dataset():
            return DualEncoderLoadData(
                interactions_path="dataset/ml-1m/ratings.dat",
                sparse_neg_sampling=True
            )
        
        result = benchmark(create_sparse_dataset)
        print(f"\n=== Sparse Benchmark ===")
        print(f"Mean: {result.stats.mean:.6f}s")
        print(f"Min:  {result.stats.min:.6f}s")
        print(f"Max:  {result.stats.max:.6f}s")
        
        assert result is not None
        assert result.stats.mean > 0
    
    def test_dense_benchmark(self, benchmark):
        """Бенчмарк dense негативного семплирования"""
        def create_dense_dataset():
            return DualEncoderLoadData(
                interactions_path="dataset/ml-1m/ratings.dat",
                sparse_neg_sampling=False
            )
        
        result = benchmark(create_dense_dataset)
        print(f"\n=== Dense Benchmark ===")
        print(f"Mean: {result.stats.mean:.6f}s")
        print(f"Min:  {result.stats.min:.6f}s")
        print(f"Max:  {result.stats.max:.6f}s")
        
        assert result is not None
        assert result.stats.mean > 0
    
    def test_memory_efficiency(self):
        """Тест эффективности использования памяти"""
        
        # Измеряем память для sparse
        process = psutil.Process(os.getpid())
        gc.collect()  # Очищаем память
        
        initial_memory = self.get_memory_usage()
        datasets_sparse = DualEncoderLoadData(
            interactions_path="dataset/ml-1m/ratings.dat",
            sparse_neg_sampling=True
        )
        sparse_memory = self.get_memory_usage() - initial_memory
        
        # Измеряем память для dense
        del datasets_sparse
        gc.collect()
        
        initial_memory = self.get_memory_usage()
        datasets_dense = DualEncoderLoadData(
            interactions_path="dataset/ml-1m/ratings.dat",
            sparse_neg_sampling=False
        )
        dense_memory = self.get_memory_usage() - initial_memory
        
        print(f"\n=== Memory Efficiency ===")
        print(f"Sparse memory: {sparse_memory:.2f} MB")
        print(f"Dense memory:  {dense_memory:.2f} MB")
        print(f"Memory saved:  {dense_memory - sparse_memory:.2f} MB")
        print(f"Efficiency:    {dense_memory/sparse_memory:.2f}x")
        
        # Sparse должен использовать меньше памяти
        assert sparse_memory < dense_memory * 0.8  # Sparse должен быть эффективнее по памяти 