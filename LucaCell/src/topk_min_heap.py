#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/6/17 16:31
@project: LucaCell
@file: max_heap
@desc: xxxx
"""
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class HeapElement:
    seq_id: str
    access_cnt: int

    def __lt__(self, other: 'HeapElement') -> bool:
        """Define comparison based on access_cnt for min heap."""
        return self.access_cnt < other.access_cnt

    def __repr__(self):
        return f"(seq_id={self.seq_id}, access_cnt={self.access_cnt})"


class TopKMinHeap:
    def __init__(self, k: int):
        """
        Initialize the TopKMinHeap.

        :param k: Maximum capacity of the heap.
        """
        self.heap: List[HeapElement] = []
        self.k = k
        self.position_map: Dict[str, int] = {}  # Maps seq_id to index in heap

    def size(self) -> int:
        """
        Get the current size of the heap.

        :return: Number of elements in the heap.
        """
        return len(self.heap)

    def delete_heap_top(self) -> HeapElement:
        """
        Remove and return the top element of the heap.

        :return: The top element (HeapElement).
        :raises IndexError: If the heap is empty.
        """
        if not self.heap:
            raise IndexError("delete_heap_top from an empty heap")
        top_element = self.heap[0]
        if self.size() > 1:
            self._swap(0, self.size() - 1)
        self.heap.pop()
        del self.position_map[top_element.seq_id]
        if self.heap:
            self._heapify_down(0)
        # print(f"Deleted heap top: {top_element}")
        return top_element

    def insert(self, seq_id: str, access_cnt: int):
        """
        Insert a new element into the heap. If the heap exceeds capacity,
        remove the heap top element.

        :param seq_id: Sequence ID of the element.
        :param access_cnt: Access count of the element.
        :raises ValueError: If the seq_id already exists.
        """
        if seq_id in self.position_map:
            raise ValueError(f"Seq ID {seq_id} already exists in the heap. Use update() instead.")

        new_element = HeapElement(seq_id, access_cnt)

        if self.size() < self.k:
            # Heap not full, push directly
            self.heap.append(new_element)
            self.position_map[seq_id] = self.size() - 1
            self._heapify_up(self.size() - 1)
            # print(f"Inserted: {new_element} | Heap: {self.heap}")
            return None
        else:
            # Heap full, compare with heap top
            removed_element = self.delete_heap_top()
            self.heap.append(new_element)
            self.position_map[seq_id] = self.size() - 1
            self._heapify_up(self.size() - 1)
            return removed_element
            # print(f"Inserted: {new_element} | Removed: {removed_element} | Heap: {self.heap}")

    def update(self, seq_id: str, new_access_cnt: int):
        """
        Update the access_cnt of an existing element and adjust the heap.

        :param seq_id: Sequence ID of the element to update.
        :param new_access_cnt: New access count.
        :raises KeyError: If the seq_id does not exist.
        """
        if seq_id not in self.position_map:
            raise KeyError(f"Seq ID {seq_id} not found in the heap.")

        index = self.position_map[seq_id]
        old_access_cnt = self.heap[index].access_cnt
        self.heap[index].access_cnt = new_access_cnt

        if new_access_cnt > old_access_cnt:
            self._heapify_down(index)
        elif new_access_cnt < old_access_cnt:
            self._heapify_up(index)
        # No action needed if access_cnt is unchanged
        # print(f"Updated seq_id={seq_id} from access_cnt={old_access_cnt} to access_cnt={new_access_cnt} | Heap: {self.heap}")

    def _heapify_up(self, index: int):
        """
        Heapify upwards to maintain heap property.

        :param index: Index of the element to heapify.
        """
        while index > 0:
            parent = self.parent(index)
            if self.heap[parent] > self.heap[index]:
                self._swap(parent, index)
                index = parent
            else:
                break

    def _heapify_down(self, index: int):
        """
        Heapify downwards to maintain heap property.

        :param index: Index of the element to heapify.
        """
        size = self.size()
        while True:
            smallest = index
            left = self.left_child(index)
            right = self.right_child(index)

            if left < size and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < size and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest != index:
                self._swap(index, smallest)
                index = smallest
            else:
                break

    def _swap(self, i: int, j: int):
        """
        Swap two elements in the heap and update their positions.

        :param i: Index of the first element.
        :param j: Index of the second element.
        """
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.position_map[self.heap[i].seq_id] = i
        self.position_map[self.heap[j].seq_id] = j
        # print(f"Swapped elements at indices {i} and {j}: {self.heap[i]}, {self.heap[j]}")

    def parent(self, index: int) -> int:
        """
        Get the parent index of a given index.

        :param index: Current index.
        :return: Parent index.
        """
        return (index - 1) // 2

    def left_child(self, index: int) -> int:
        """
        Get the left child index of a given index.

        :param index: Current index.
        :return: Left child index.
        """
        return 2 * index + 1

    def right_child(self, index: int) -> int:
        """
        Get the right child index of a given index.

        :param index: Current index.
        :return: Right child index.
        """
        return 2 * index + 2

    def get_access_cnt(self, seq_id: str):
        return self.heap[self.position_map[seq_id]].access_cnt

    def __str__(self):
        """
        Return a string representation of the heap.

        :return: String representation.
        """
        return str(self.heap)


if __name__ == "__main__":
    # 初始化一个容量为 5 的小顶堆
    k = 5
    heap = TopKMinHeap(k=k)

    # 插入一些元素
    elements_to_insert = [
        (1, 10),
        (2, 20),
        (3, 15),
        (4, 30),
        (5, 25),
        (6, 35),  # 超过容量，应该删除堆顶
        (7, 5),    # 不插入，因为 access_cnt <= 堆顶
    ]

    for seq_id, access_cnt in elements_to_insert:
        print(f"\nInserting (seq_id={seq_id}, access_cnt={access_cnt}):")
        try:
            heap.insert(seq_id, access_cnt)
        except ValueError as ve:
            print(f"Error: {ve}")
        print(f"Heap after insertion: {heap}")

    # 更新元素的访问次数
    updates = [
        (2, 5),    # 减少 access_cnt
        (4, 40),   # 增加 access_cnt
        (5, 15),   # 减少 access_cnt
        (6, 10),   # 减少 access_cnt
        (1, 50),   # 尝试更新不存在的 seq_id=1
    ]

    for seq_id, new_access_cnt in updates:
        print(f"\nUpdating (seq_id={seq_id}) to new_access_cnt={new_access_cnt}:")
        try:
            heap.update(seq_id, new_access_cnt)
        except KeyError as ke:
            print(f"Error: {ke}")
        print(f"Heap after update: {heap}")

    # 查看堆的大小
    print(f"\nCurrent heap size: {heap.size()}")

    # 删除堆顶元素
    try:
        top_element = heap.delete_heap_top()
        print(f"\nDeleted heap top: {top_element}")
    except IndexError as ie:
        print(f"Error: {ie}")
    print(f"Heap after deleting top: {heap}")

    # 再次查看堆的大小
    print(f"\nCurrent heap size: {heap.size()}")

    # 尝试插入一个新的元素
    new_insert = (8, 50)
    print(f"\nInserting (seq_id={new_insert[0]}, access_cnt={new_insert[1]}):")
    try:
        heap.insert(*new_insert)
    except ValueError as ve:
        print(f"Error: {ve}")
    print(f"Heap after insertion: {heap}")

    # 尝试更新一个不存在的 seq_id
    non_existing_update = (9, 60)
    print(f"\nUpdating (seq_id={non_existing_update[0]}) to new_access_cnt={non_existing_update[1]}:")
    try:
        heap.update(*non_existing_update)
    except KeyError as ke:
        print(f"Error: {ke}")
    print(f"Heap after update: {heap}")
