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
}

impl<K, V> SparseCountVec<K, V>
where
    K: Copy + Ord,
    V: Copy + Zero,
{
    fn new() -> Self {
        let root = LeafNode::new();
        SparseCountVec {
            leaf_arena: vec![root],
            internal_arena: Vec::new(),
            root: NodePtr::Leaf(0),
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
    {
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

        while let Some((internal_idx, child_pos)) = path.pop() {
            let node = self.internal_mut(internal_idx);
            if node.children.len() == INTERNAL_SIZE {
                // split internal node
                let mid = node.keys.len() / 2;
                let promote_key = node.keys[mid];

                let right_keys: SmallVec<[_; INTERNAL_SIZE - 1]> =
                    node.keys.drain(mid + 1..).collect();
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
                new_child = NodePtr::Internal(right_idx);
            } else {
                // insert into internal node and finish
                node.keys.insert(child_pos, new_key);
                node.children.insert(child_pos + 1, new_child);
                break;
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
