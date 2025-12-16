use num::traits::Zero;
use smallvec::SmallVec;
use std::cmp::Ord;

const INTERNAL_SIZE: usize = 8;
const LEAF_SIZE: usize = 8;

type LeafIdx = u32;
type InternalIdx = u32;
const NULL_IDX: u32 = u32::MAX;

#[derive(Clone, Copy)]
enum NodePtr {
    Internal(InternalIdx),
    Leaf(LeafIdx),
}

// This is mutable sparse vector implemeted as a B+ tree.
struct SparseCountVec<K, V> {
    leaf_arena: Vec<LeafNode<K, V>>,
    internal_arena: Vec<InternalNode<K>>,
    root: NodePtr,
    len: usize,
}

impl<K, V> SparseCountVec<K, V>
where
    K: Copy + Ord,
    V: Copy + Zero,
{
    fn new(len: usize) -> Self {
        let root = LeafNode::new();
        SparseCountVec {
            leaf_arena: vec![root],
            internal_arena: Vec::new(),
            root: NodePtr::Leaf(0),
            len,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    // Create an iterator over non-zero key-value pairs
    fn iter(&self) -> SparseCountVecIter<K, V> {
        // Find the leftmost leaf
        let mut current = self.root;
        let first_leaf = loop {
            match current {
                NodePtr::Leaf(idx) => break idx,
                NodePtr::Internal(idx) => {
                    let internal = self.internal(idx);
                    current = internal.children[0];
                }
            }
        };

        SparseCountVecIter {
            vec: self,
            current_leaf: first_leaf,
            position_in_leaf: 0,
        }
    }

    // Create an iterator over all values (including implicit zeros)
    fn iter_dense(&self) -> DenseIter<K, V>
    where
        K: TryFrom<usize>,
    {
        // Find the leftmost leaf
        let mut current = self.root;
        let first_leaf = loop {
            match current {
                NodePtr::Leaf(idx) => break idx,
                NodePtr::Internal(idx) => {
                    let internal = self.internal(idx);
                    current = internal.children[0];
                }
            }
        };

        DenseIter {
            vec: self,
            current_index: 0,
            current_leaf: first_leaf,
            position_in_leaf: 0,
        }
    }

    fn new_leaf(&mut self, leaf: LeafNode<K, V>) -> LeafIdx {
        self.leaf_arena.push(leaf);
        self.leaf_arena.len() as LeafIdx - 1
    }

    fn new_internal(&mut self, internal: InternalNode<K>) -> InternalIdx {
        self.internal_arena.push(internal);
        self.internal_arena.len() as InternalIdx - 1
    }

    fn leaf(&self, idx: LeafIdx) -> &LeafNode<K, V> {
        &self.leaf_arena[idx as usize]
    }

    fn leaf_mut(&mut self, idx: LeafIdx) -> &mut LeafNode<K, V> {
        &mut self.leaf_arena[idx as usize]
    }

    fn internal(&self, idx: InternalIdx) -> &InternalNode<K> {
        &self.internal_arena[idx as usize]
    }

    fn internal_mut(&mut self, idx: InternalIdx) -> &mut InternalNode<K> {
        &mut self.internal_arena[idx as usize]
    }

    fn find_leaf(&self, key: K) -> LeafIdx {
        let mut current = self.root;
        loop {
            match current {
                NodePtr::Leaf(idx) => return idx,
                NodePtr::Internal(idx) => {
                    let internal = &self.internal_arena[idx as usize];
                    // since we plan to use a large leaf size, there will typically be a smaller number of leaves, so linear search
                    // is likely faster.
                    // let pos = internal.keys.partition_point(|&k| k < key);
                    let pos = internal
                        .keys
                        .iter()
                        .position(|&k| key < k)
                        .unwrap_or(internal.keys.len());
                    current = internal.children[pos];
                }
            }
        }
    }

    // This is the only function we use to modify the structure. It updates
    // a mutable value with the given function, inserting a zero first if
    // the key is not already present.
    fn update_count<F>(&mut self, key: K, update_fn: F)
    where
        F: FnOnce(&mut V),
        K: TryInto<usize>,
    {
        // Bounds check: ensure key is within [0, len)
        let key_usize = key.try_into().ok().expect("Key conversion failed");
        assert!(
            key_usize < self.len,
            "Key {} is out of bounds (len = {})",
            key_usize,
            self.len
        );

        let leaf_idx = self.find_leaf(key);
        let leaf = self.leaf_mut(leaf_idx);

        match leaf.binary_search(key) {
            Ok(pos) => {
                update_fn(&mut leaf.keyvals[pos].1);
                return;
            }
            Err(pos) => {
                let mut val = V::zero();
                update_fn(&mut val);
                if leaf.keyvals.len() < LEAF_SIZE {
                    leaf.keyvals.insert(pos, (key, val));
                    return;
                }

                self.insert_splitting(key, val);
            }
        }
    }

    fn insert_splitting(&mut self, key: K, val: V) {
        // trace path down to leaf node
        let mut path: SmallVec<[(InternalIdx, usize); 16]> = SmallVec::new();
        let mut ptr = self.root;

        let leaf_idx = loop {
            match ptr {
                NodePtr::Leaf(idx) => break idx,
                NodePtr::Internal(idx) => {
                    let node = self.internal(idx);
                    let pos = node
                        .keys
                        .iter()
                        .position(|&k| key < k)
                        .unwrap_or(node.keys.len());
                    path.push((idx, pos));
                    ptr = node.children[pos];
                }
            }
        };

        let leaf = self.leaf_mut(leaf_idx);

        assert!(leaf.keyvals.len() == LEAF_SIZE);
        let mid = LEAF_SIZE / 2;
        let split_key = leaf.keyvals[mid].0;

        let right_keyvals: SmallVec<[_; LEAF_SIZE]> = leaf.keyvals.drain(mid..).collect();
        let right = LeafNode {
            keyvals: right_keyvals,
            sibling: leaf.sibling,
        };
        let right_idx = self.new_leaf(right);
        self.leaf_mut(leaf_idx).sibling = right_idx;

        if key < split_key {
            let left = self.leaf_mut(leaf_idx);
            let insert_pos = left.binary_search(key).unwrap_err();
            left.keyvals.insert(insert_pos, (key, val));
        } else {
            let right = self.leaf_mut(right_idx);
            let insert_pos = right.binary_search(key).unwrap_err();
            right.keyvals.insert(insert_pos, (key, val));
        }

        // progogate split upwards
        let mut new_child = NodePtr::Leaf(right_idx);
        let mut new_key = split_key;
        let mut old_child = NodePtr::Leaf(leaf_idx);

        while let Some((internal_idx, child_pos)) = path.pop() {
            let node = self.internal_mut(internal_idx);
            if node.children.len() == INTERNAL_SIZE {
                // split internal node
                let mid = node.keys.len() / 2;
                let promote_key = node.keys[mid];

                let right_keys: SmallVec<[_; INTERNAL_SIZE - 1]> =
                    node.keys.drain(mid + 1..).collect();
                node.keys.pop(); // Remove the promoted key from left node
                let right_children: SmallVec<[_; INTERNAL_SIZE]> =
                    node.children.drain(mid + 1..).collect();

                let right = InternalNode {
                    keys: right_keys,
                    children: right_children,
                };
                let right_idx = self.new_internal(right);

                if child_pos <= mid {
                    let left = self.internal_mut(internal_idx);
                    left.keys.insert(child_pos, new_key);
                    left.children.insert(child_pos + 1, new_child);
                } else {
                    let right = self.internal_mut(right_idx);
                    let right_pos = child_pos - mid - 1;
                    right.keys.insert(right_pos, new_key);
                    right.children.insert(right_pos + 1, new_child);
                }
                new_key = promote_key;
                old_child = NodePtr::Internal(internal_idx);
                new_child = NodePtr::Internal(right_idx);
            } else {
                // insert into internal node and finish
                node.keys.insert(child_pos, new_key);
                node.children.insert(child_pos + 1, new_child);
                return;
            }
        }

        // If we get here, we've split all the way to the root
        // Create a new root internal node
        let new_root = InternalNode {
            keys: SmallVec::from_slice(&[new_key]),
            children: SmallVec::from_slice(&[old_child, new_child]),
        };
        let new_root_idx = self.new_internal(new_root);
        self.root = NodePtr::Internal(new_root_idx);
    }
}

// Iterator for traversing non-zero key-value pairs
pub struct SparseCountVecIter<'a, K, V> {
    vec: &'a SparseCountVec<K, V>,
    current_leaf: LeafIdx,
    position_in_leaf: usize,
}

impl<'a, K, V> Iterator for SparseCountVecIter<'a, K, V>
where
    K: Copy + Ord,
    V: Copy + Zero + PartialEq,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Check if we've reached the end
            if self.current_leaf == NULL_IDX {
                return None;
            }

            let leaf = self.vec.leaf(self.current_leaf);

            // Try to get the next item from the current leaf
            while self.position_in_leaf < leaf.keyvals.len() {
                let (key, val) = leaf.keyvals[self.position_in_leaf];
                self.position_in_leaf += 1;

                // Skip zero values (both explicit zeros in the tree)
                if !val.is_zero() {
                    return Some((key, val));
                }
            }

            // Move to the next leaf
            self.current_leaf = leaf.sibling;
            self.position_in_leaf = 0;
        }
    }
}

// Iterator for traversing all values including implicit zeros
pub struct DenseIter<'a, K, V> {
    vec: &'a SparseCountVec<K, V>,
    current_index: usize,
    current_leaf: LeafIdx,
    position_in_leaf: usize,
}

impl<'a, K, V> Iterator for DenseIter<'a, K, V>
where
    K: Copy + Ord + TryFrom<usize>,
    V: Copy + Zero,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've exhausted all indices
        if self.current_index >= self.vec.len {
            return None;
        }

        // Convert current index to key
        let key = K::try_from(self.current_index).ok()?;
        self.current_index += 1;

        // Navigate to the correct leaf if needed
        if self.current_leaf == NULL_IDX {
            return Some(V::zero());
        }

        let leaf = self.vec.leaf(self.current_leaf);

        // Check if the current position in leaf has our key
        if self.position_in_leaf < leaf.keyvals.len() {
            let (leaf_key, leaf_val) = leaf.keyvals[self.position_in_leaf];
            if leaf_key == key {
                // Found the key, advance position
                self.position_in_leaf += 1;

                // Move to next leaf if we've exhausted this one
                if self.position_in_leaf >= leaf.keyvals.len() {
                    self.current_leaf = leaf.sibling;
                    self.position_in_leaf = 0;
                }

                return Some(leaf_val);
            }
        }

        // Key not in this position, might be in next leaf or doesn't exist
        // We need to check if we should move to the next leaf
        loop {
            if self.current_leaf == NULL_IDX {
                // No more leaves, return zero
                return Some(V::zero());
            }

            let leaf = self.vec.leaf(self.current_leaf);

            if self.position_in_leaf >= leaf.keyvals.len() {
                // Exhausted current leaf, move to next
                self.current_leaf = leaf.sibling;
                self.position_in_leaf = 0;
                continue;
            }

            let (leaf_key, leaf_val) = leaf.keyvals[self.position_in_leaf];

            if leaf_key == key {
                // Found it
                self.position_in_leaf += 1;

                // Move to next leaf if exhausted
                if self.position_in_leaf >= leaf.keyvals.len() {
                    self.current_leaf = leaf.sibling;
                    self.position_in_leaf = 0;
                }

                return Some(leaf_val);
            } else if leaf_key > key {
                // Key doesn't exist (we've passed where it would be)
                return Some(V::zero());
            } else {
                // This shouldn't happen if we're iterating sequentially
                // but return zero to be safe
                return Some(V::zero());
            }
        }
    }
}

struct InternalNode<K> {
    keys: SmallVec<[K; INTERNAL_SIZE - 1]>,
    children: SmallVec<[NodePtr; INTERNAL_SIZE]>,
}

impl<K> InternalNode<K>
where
    K: Copy + Ord,
{
    fn new() -> Self {
        InternalNode {
            keys: SmallVec::new(),
            children: SmallVec::new(),
        }
    }

    fn binary_search(&self, key: K) -> Result<usize, usize> {
        self.keys.binary_search(&key)
    }
}

struct LeafNode<K, V> {
    keyvals: SmallVec<[(K, V); LEAF_SIZE]>,
    sibling: LeafIdx,
}

impl<K, V> LeafNode<K, V>
where
    K: Copy + Ord,
{
    fn new() -> Self {
        LeafNode {
            keyvals: SmallVec::new(),
            sibling: NULL_IDX,
        }
    }

    fn binary_search(&self, key: K) -> Result<usize, usize> {
        self.keyvals.binary_search_by_key(&key, |&(k, _)| k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to verify the tree structure is valid
    impl<K, V> SparseCountVec<K, V>
    where
        K: Copy + Ord + std::fmt::Debug,
        V: Copy + Zero + std::fmt::Debug,
    {
        fn get(&self, key: K) -> Option<V> {
            let leaf_idx = self.find_leaf(key);
            let leaf = self.leaf(leaf_idx);
            match leaf.binary_search(key) {
                Ok(pos) => Some(leaf.keyvals[pos].1),
                Err(_) => None,
            }
        }

        // Collect all key-value pairs in sorted order by traversing leaf siblings
        fn collect_all(&self) -> Vec<(K, V)> {
            let mut result = Vec::new();

            // Find the leftmost leaf
            let mut current = self.root;
            loop {
                match current {
                    NodePtr::Leaf(idx) => {
                        let mut leaf_idx = idx;
                        loop {
                            let leaf = self.leaf(leaf_idx);
                            result.extend_from_slice(&leaf.keyvals);
                            if leaf.sibling == NULL_IDX {
                                break;
                            }
                            leaf_idx = leaf.sibling;
                        }
                        return result;
                    }
                    NodePtr::Internal(idx) => {
                        let internal = self.internal(idx);
                        current = internal.children[0];
                    }
                }
            }
        }

        // Verify the tree maintains sorted order
        fn verify_sorted(&self) -> bool {
            let all = self.collect_all();
            all.windows(2).all(|w| w[0].0 < w[1].0)
        }
    }

    #[test]
    fn test_new_sparse_vec() {
        let vec: SparseCountVec<u32, i32> = SparseCountVec::new(100);
        assert_eq!(vec.leaf_arena.len(), 1);
        assert_eq!(vec.internal_arena.len(), 0);
        assert_eq!(vec.len(), 100);
        matches!(vec.root, NodePtr::Leaf(0));
    }

    #[test]
    fn test_single_insert() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);
        vec.update_count(5, |v| *v += 10);

        assert_eq!(vec.get(5), Some(10));
        assert_eq!(vec.get(3), None);
        assert_eq!(vec.get(7), None);
    }

    #[test]
    fn test_multiple_inserts_no_split() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert fewer items than LEAF_SIZE
        for i in 0..LEAF_SIZE - 1 {
            vec.update_count(i as u32, |v| *v += i as i32);
        }

        // Verify all values
        for i in 0..LEAF_SIZE - 1 {
            assert_eq!(vec.get(i as u32), Some(i as i32));
        }

        // Should still be a single leaf node
        assert_eq!(vec.leaf_arena.len(), 1);
        assert_eq!(vec.internal_arena.len(), 0);
    }

    #[test]
    fn test_update_existing_key() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        vec.update_count(5, |v| *v += 10);
        assert_eq!(vec.get(5), Some(10));

        vec.update_count(5, |v| *v += 5);
        assert_eq!(vec.get(5), Some(15));

        vec.update_count(5, |v| *v -= 3);
        assert_eq!(vec.get(5), Some(12));
    }

    #[test]
    fn test_insert_sorted_order() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert in ascending order
        for i in 0..20 {
            vec.update_count(i, |v| *v += i as i32);
        }

        // Verify all values
        for i in 0..20 {
            assert_eq!(vec.get(i), Some(i as i32), "Failed at key {}", i);
        }

        // Verify sorted order
        assert!(vec.verify_sorted());
    }

    #[test]
    fn test_insert_reverse_order() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert in descending order
        for i in (0..20).rev() {
            vec.update_count(i, |v| *v += i as i32);
        }

        // Verify all values
        for i in 0..20 {
            assert_eq!(vec.get(i), Some(i as i32), "Failed at key {}", i);
        }

        // Verify sorted order
        assert!(vec.verify_sorted());
    }

    #[test]
    fn test_insert_random_order() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert in a pseudo-random order
        let keys = vec![15, 3, 8, 1, 12, 6, 18, 4, 10, 2, 16, 7, 14, 5, 11, 9, 13, 0, 17, 19];
        for &key in &keys {
            vec.update_count(key, |v| *v += key as i32);
        }

        // Verify all values
        for i in 0..20 {
            assert_eq!(vec.get(i), Some(i as i32), "Failed at key {}", i);
        }

        // Verify sorted order
        assert!(vec.verify_sorted());
    }

    #[test]
    fn test_leaf_split() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert exactly LEAF_SIZE items (no split yet)
        for i in 0..LEAF_SIZE {
            vec.update_count(i as u32, |v| *v += i as i32);
        }
        assert_eq!(vec.leaf_arena.len(), 1);

        // Insert one more to trigger split
        vec.update_count(LEAF_SIZE as u32, |v| *v += LEAF_SIZE as i32);

        // Should now have 2 leaf nodes
        assert_eq!(vec.leaf_arena.len(), 2);

        // Verify all values are still accessible
        for i in 0..=LEAF_SIZE {
            assert_eq!(vec.get(i as u32), Some(i as i32));
        }

        // Verify sorted order
        assert!(vec.verify_sorted());
    }

    #[test]
    fn test_multiple_leaf_splits() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert enough items to trigger multiple splits
        let n = LEAF_SIZE * 5;
        for i in 0..n {
            vec.update_count(i as u32, |v| *v += i as i32);
        }

        // Verify all values
        for i in 0..n {
            assert_eq!(vec.get(i as u32), Some(i as i32), "Failed at key {}", i);
        }

        // Verify sorted order
        assert!(vec.verify_sorted());
    }

    #[test]
    fn test_internal_node_creation() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert enough to create internal nodes
        let n = LEAF_SIZE * 3;
        for i in 0..n {
            vec.update_count(i as u32, |v| *v += i as i32);
        }

        // Should have created at least one internal node
        assert!(vec.internal_arena.len() > 0);

        // Verify all values
        for i in 0..n {
            assert_eq!(vec.get(i as u32), Some(i as i32));
        }
    }

    #[test]
    fn test_large_tree() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert a large number of items
        let n = 1000;
        for i in 0..n {
            vec.update_count(i, |v| *v += i as i32);
        }

        // Verify all values
        for i in 0..n {
            assert_eq!(vec.get(i), Some(i as i32));
        }

        // Verify sorted order
        assert!(vec.verify_sorted());
    }

    #[test]
    fn test_interleaved_inserts_and_updates() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert some initial values
        for i in 0..10 {
            vec.update_count(i * 2, |v| *v += i as i32);
        }

        // Update some existing keys
        for i in 0..5 {
            vec.update_count(i * 2, |v| *v *= 2);
        }

        // Insert some new keys in between
        for i in 0..10 {
            vec.update_count(i * 2 + 1, |v| *v += 100);
        }

        // Verify even keys
        for i in 0..5 {
            assert_eq!(vec.get(i * 2), Some((i * 2) as i32));
        }
        for i in 5..10 {
            assert_eq!(vec.get(i * 2), Some(i as i32));
        }

        // Verify odd keys
        for i in 0..10 {
            assert_eq!(vec.get(i * 2 + 1), Some(100));
        }

        // Verify sorted order
        assert!(vec.verify_sorted());
    }

    #[test]
    fn test_zero_initialization() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Update should start with zero
        vec.update_count(5, |v| {
            assert_eq!(*v, 0);
            *v += 10;
        });

        assert_eq!(vec.get(5), Some(10));
    }

    #[test]
    fn test_different_value_types() {
        // Test with f32
        let mut vec_f32: SparseCountVec<u32, f32> = SparseCountVec::new(1000);
        vec_f32.update_count(1, |v| *v += 1.5);
        vec_f32.update_count(1, |v| *v += 2.5);
        assert_eq!(vec_f32.get(1), Some(4.0));

        // Test with u64
        let mut vec_u64: SparseCountVec<u32, u64> = SparseCountVec::new(1000);
        vec_u64.update_count(1, |v| *v += 1000);
        assert_eq!(vec_u64.get(1), Some(1000));
    }

    #[test]
    fn test_collect_all_empty() {
        let vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);
        let all = vec.collect_all();
        assert_eq!(all.len(), 0);
    }

    #[test]
    fn test_collect_all_with_items() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        let keys = vec![5, 2, 8, 1, 9, 3];
        for &key in &keys {
            vec.update_count(key, |v| *v += key as i32);
        }

        let all = vec.collect_all();

        // Should be in sorted order
        assert_eq!(all, vec![(1, 1), (2, 2), (3, 3), (5, 5), (8, 8), (9, 9)]);
    }

    #[test]
    fn test_sibling_links_after_splits() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert enough to trigger multiple splits
        for i in 0..30 {
            vec.update_count(i, |v| *v += i as i32);
        }

        // Collect all should traverse siblings correctly
        let all = vec.collect_all();
        assert_eq!(all.len(), 30);

        // Verify they're in order
        for (i, &(k, v)) in all.iter().enumerate() {
            assert_eq!(k, i as u32);
            assert_eq!(v, i as i32);
        }
    }

    #[test]
    fn test_sparse_keys() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(2000000);

        // Insert with large gaps between keys
        let keys = vec![10, 1000, 10000, 100000, 1000000];
        for &key in &keys {
            vec.update_count(key, |v| *v += 1);
        }

        // Verify all sparse keys
        for &key in &keys {
            assert_eq!(vec.get(key), Some(1));
        }

        // Verify keys in between don't exist
        assert_eq!(vec.get(500), None);
        assert_eq!(vec.get(50000), None);

        // Verify sorted order
        assert!(vec.verify_sorted());
    }

    // Iterator tests
    #[test]
    fn test_iter_empty() {
        let vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);
        let items: Vec<_> = vec.iter().collect();
        assert_eq!(items.len(), 0);
    }

    #[test]
    fn test_iter_single_element() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);
        vec.update_count(5, |v| *v += 10);

        let items: Vec<_> = vec.iter().collect();
        assert_eq!(items, vec![(5, 10)]);
    }

    #[test]
    fn test_iter_multiple_elements() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        for i in 0..10 {
            vec.update_count(i, |v| *v += i as i32);
        }

        let items: Vec<_> = vec.iter().collect();
        // Note: key 0 has value 0, so it should be filtered out
        assert_eq!(
            items,
            vec![
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9)
            ]
        );
    }

    #[test]
    fn test_iter_skips_explicit_zeros() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert some values
        for i in 0..10 {
            vec.update_count(i, |v| *v += i as i32 + 1);
        }

        // Set some values to zero
        vec.update_count(2, |v| *v = 0);
        vec.update_count(5, |v| *v = 0);
        vec.update_count(7, |v| *v = 0);

        let items: Vec<_> = vec.iter().collect();
        // Should skip keys 2, 5, and 7
        assert_eq!(
            items,
            vec![(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (8, 9), (9, 10)]
        );
    }

    #[test]
    fn test_iter_after_splits() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert enough to trigger multiple splits
        for i in 0..50 {
            vec.update_count(i, |v| *v += (i + 1) as i32);
        }

        let items: Vec<_> = vec.iter().collect();
        assert_eq!(items.len(), 50);

        // Verify order and values
        for (i, &(key, val)) in items.iter().enumerate() {
            assert_eq!(key, i as u32);
            assert_eq!(val, (i + 1) as i32);
        }
    }

    #[test]
    fn test_iter_sorted_order() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert in random order
        let keys = vec![15, 3, 8, 1, 12, 6, 18, 4, 10, 2, 16, 7, 14, 5, 11, 9, 13, 17, 19];
        for &key in &keys {
            vec.update_count(key, |v| *v += key as i32);
        }

        let items: Vec<_> = vec.iter().collect();

        // Verify iterator returns items in sorted order
        for i in 1..items.len() {
            assert!(items[i - 1].0 < items[i].0);
        }

        // Verify we got all expected items
        let expected: Vec<_> = keys.iter().map(|&k| (k, k as i32)).collect();
        let mut sorted_expected = expected.clone();
        sorted_expected.sort_by_key(|&(k, _)| k);

        assert_eq!(items, sorted_expected);
    }

    #[test]
    fn test_iter_with_sparse_keys() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(200000);

        // Insert with large gaps
        let keys = vec![10, 1000, 10000, 100000];
        for &key in &keys {
            vec.update_count(key, |v| *v += key as i32);
        }

        let items: Vec<_> = vec.iter().collect();
        assert_eq!(
            items,
            vec![
                (10, 10),
                (1000, 1000),
                (10000, 10000),
                (100000, 100000)
            ]
        );
    }

    #[test]
    fn test_iter_all_zeros() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert some values then set them all to zero
        for i in 0..10 {
            vec.update_count(i, |v| *v += 10);
        }

        for i in 0..10 {
            vec.update_count(i, |v| *v = 0);
        }

        let items: Vec<_> = vec.iter().collect();
        assert_eq!(items.len(), 0);
    }

    #[test]
    fn test_iter_mixed_zeros_and_nonzeros() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Alternating zero and non-zero values
        for i in 0..20 {
            if i % 2 == 0 {
                vec.update_count(i, |v| *v += i as i32);
            } else {
                vec.update_count(i, |v| *v += 0); // Explicitly add zero
            }
        }

        let items: Vec<_> = vec.iter().collect();

        // Should only get even keys (odd keys have zero values)
        // Note: key 0 also has value 0, so it should be filtered out too
        let expected: Vec<_> = (0..20)
            .filter(|i| i % 2 == 0 && *i != 0)
            .map(|i| (i, i as i32))
            .collect();

        assert_eq!(items, expected);
    }

    #[test]
    fn test_iter_reusable() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        for i in 1..=5 {
            vec.update_count(i, |v| *v += i as i32);
        }

        // First iteration
        let items1: Vec<_> = vec.iter().collect();
        // Second iteration on the same tree
        let items2: Vec<_> = vec.iter().collect();

        assert_eq!(items1, items2);
        assert_eq!(items1, vec![(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]);
    }

    // Bounds checking tests
    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_bounds_check_fails() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(10);
        vec.update_count(10, |v| *v += 1); // Should panic: 10 is out of bounds [0, 10)
    }

    #[test]
    fn test_bounds_check_passes() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(10);
        vec.update_count(9, |v| *v += 1); // Should work: 9 is within [0, 10)
        assert_eq!(vec.get(9), Some(1));
    }

    // Dense iterator tests
    #[test]
    fn test_iter_dense_empty() {
        let vec: SparseCountVec<u32, i32> = SparseCountVec::new(0);
        let items: Vec<_> = vec.iter_dense().collect();
        assert_eq!(items.len(), 0);
    }

    #[test]
    fn test_iter_dense_all_zeros() {
        let vec: SparseCountVec<u32, i32> = SparseCountVec::new(5);
        let items: Vec<_> = vec.iter_dense().collect();
        assert_eq!(items, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_iter_dense_single_value() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(10);
        vec.update_count(5, |v| *v += 42);

        let items: Vec<_> = vec.iter_dense().collect();
        assert_eq!(items, vec![0, 0, 0, 0, 0, 42, 0, 0, 0, 0]);
    }

    #[test]
    fn test_iter_dense_multiple_values() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(10);
        vec.update_count(2, |v| *v += 2);
        vec.update_count(5, |v| *v += 5);
        vec.update_count(7, |v| *v += 7);

        let items: Vec<_> = vec.iter_dense().collect();
        assert_eq!(items, vec![0, 0, 2, 0, 0, 5, 0, 7, 0, 0]);
    }

    #[test]
    fn test_iter_dense_all_values() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(10);
        for i in 0..10 {
            vec.update_count(i, |v| *v += (i + 1) as i32);
        }

        let items: Vec<_> = vec.iter_dense().collect();
        assert_eq!(items, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_iter_dense_with_explicit_zeros() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(10);
        vec.update_count(2, |v| *v += 5);
        vec.update_count(5, |v| *v += 10);
        vec.update_count(7, |v| *v += 15);

        // Now set some to zero
        vec.update_count(5, |v| *v = 0);

        let items: Vec<_> = vec.iter_dense().collect();
        assert_eq!(items, vec![0, 0, 5, 0, 0, 0, 0, 15, 0, 0]);
    }

    #[test]
    fn test_iter_dense_after_splits() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(50);

        // Insert enough to trigger splits
        for i in 0..50 {
            if i % 3 == 0 {
                vec.update_count(i, |v| *v += i as i32);
            }
        }

        let items: Vec<_> = vec.iter_dense().collect();
        assert_eq!(items.len(), 50);

        // Verify values
        for (idx, &val) in items.iter().enumerate() {
            if idx % 3 == 0 {
                assert_eq!(val, idx as i32);
            } else {
                assert_eq!(val, 0);
            }
        }
    }

    #[test]
    fn test_iter_dense_large() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(1000);

        // Insert sparse values
        for i in (0..1000).step_by(10) {
            vec.update_count(i, |v| *v += i as i32);
        }

        let items: Vec<_> = vec.iter_dense().collect();
        assert_eq!(items.len(), 1000);

        // Verify values
        for (idx, &val) in items.iter().enumerate() {
            if idx % 10 == 0 {
                assert_eq!(val, idx as i32);
            } else {
                assert_eq!(val, 0);
            }
        }
    }

    #[test]
    fn test_iter_dense_reusable() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(5);
        vec.update_count(1, |v| *v += 10);
        vec.update_count(3, |v| *v += 30);

        let items1: Vec<_> = vec.iter_dense().collect();
        let items2: Vec<_> = vec.iter_dense().collect();

        assert_eq!(items1, items2);
        assert_eq!(items1, vec![0, 10, 0, 30, 0]);
    }

    #[test]
    fn test_iter_dense_trailing_zeros() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(20);

        // Only set first few values
        vec.update_count(0, |v| *v += 1);
        vec.update_count(1, |v| *v += 2);
        vec.update_count(2, |v| *v += 3);

        let items: Vec<_> = vec.iter_dense().collect();
        assert_eq!(items.len(), 20);
        assert_eq!(&items[0..3], &[1, 2, 3]);
        assert_eq!(&items[3..], &vec![0; 17][..]);
    }

    #[test]
    fn test_both_iterators() {
        let mut vec: SparseCountVec<u32, i32> = SparseCountVec::new(10);
        vec.update_count(2, |v| *v += 2);
        vec.update_count(5, |v| *v += 5);
        vec.update_count(7, |v| *v += 7);

        // Sparse iterator should only return non-zero values
        let sparse: Vec<_> = vec.iter().collect();
        assert_eq!(sparse, vec![(2, 2), (5, 5), (7, 7)]);

        // Dense iterator should return all values
        let dense: Vec<_> = vec.iter_dense().collect();
        assert_eq!(dense, vec![0, 0, 2, 0, 0, 5, 0, 7, 0, 0]);
    }
}
