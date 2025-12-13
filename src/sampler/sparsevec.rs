use bumpalo::Bump;
use num::traits::Zero;
use smallvec::SmallVec;
use std::cell::Cell;
use std::cmp::PartialOrd;
use std::mem::replace;
use std::ops::DerefMut;
use typed_arena::Arena;

const INTERNAL_SIZE: usize = 8;
const LEAF_SIZE: usize = 8;

// This is mutable sparse vector implemeted as a B+ tree.
struct SparseCountVec<'a, K, V> {
    bump: Bump,
    root: NodeRef<'a, K, V>,
}

impl<'a, K, V> SparseCountVec<'a, K, V> {
    fn new() -> Self {
        let bump = Bump::new();
        let root = NodeRef::None;
        SparseCountVec { bump, root }
    }
}

impl<'a, K, V> SparseCountVec<'a, K, V>
where
    K: Copy + PartialOrd,
    V: Copy + Zero,
{
    // This is the only function we use to modify the structure. It updates
    // a mutable value with the given function, inserting a zero first if
    // the key is not already present.
    fn update_count<F>(&'a mut self, key: K, update_fn: F)
    where
        F: FnOnce(&mut V),
    {
        if self.root.is_none() {
            self.root = NodeRef::LeafRef(self.bump.alloc(LeafNode::new()));
        }

        self.root.update_count(&self.bump, key, update_fn);
    }
}

enum NodeRef<'a, K, V> {
    None,
    InternalRef(&'a mut InternalNode<'a, K, V>),
    LeafRef(&'a mut LeafNode<'a, K, V>),
}

impl<'a, K, V> NodeRef<'a, K, V>
where
    K: PartialOrd,
    V: Zero,
{
    fn is_none(&self) -> bool {
        match self {
            NodeRef::None => true,
            _ => false,
        }
    }

    fn update_count(&mut self, bump: &'a Bump, key: K, update_fn: impl FnOnce(&mut V)) {
        match self {
            NodeRef::InternalRef(node) => node.update_count(bump, key, update_fn),
            NodeRef::LeafRef(node) => node.update_count(bump, key, update_fn),
            NodeRef::None => panic!("NodeRef::None should not be updated"),
        }
    }
}

struct InternalNode<'a, K, V> {
    // TODO: Do I even need a parent here?
    keys: SmallVec<[K; INTERNAL_SIZE - 1]>,
    children: SmallVec<[LeafNode<'a, K, V>; INTERNAL_SIZE]>,
}

impl<'a, K, V> InternalNode<'a, K, V>
where
    K: PartialOrd,
    V: Zero,
{
    fn update_count(&mut self, arena: &'a Bump, key: K, update_fn: impl FnOnce(&mut V)) {
        todo!();
    }
}

struct LeafNode<'a, K, V> {
    // TODO: Do I even need a parent here?
    keyvals: SmallVec<[(K, V); LEAF_SIZE]>,
    sibling: Option<&'a Self>,
}

impl<'a, K, V> LeafNode<'a, K, V>
where
    K: Copy + PartialOrd,
    V: Copy + Zero,
{
    fn new() -> Self {
        LeafNode {
            keyvals: SmallVec::new(),
            sibling: None,
        }
    }

    // Update a count vector. If a new node was created in the process, return a reference to it.
    fn update_count(
        &mut self,
        bump: &'a Bump,
        key: K,
        update_fn: impl FnOnce(&mut V),
    ) -> NodeRef<'a, K, V> {
        // figure out where the key goes with a linear scan, updating in place if it's present
        let mut place = None;
        for (i, (k, v)) in self.keyvals.iter_mut().enumerate() {
            if *k == key {
                update_fn(v);
                return NodeRef::None;
            } else if *k > key {
                place = Some(i);
            }
        }

        if place.is_none() {
            place = Some(self.keyvals.len());
        }
        let place = place.unwrap();

        // split the node if it's full
        if self.keyvals.len() >= LEAF_SIZE {
            let right = bump.alloc(Self::new());

            let mid = self.keyvals.len() / 2;
            right.keyvals.extend_from_slice(&self.keyvals[mid..]);
            self.keyvals.truncate(mid);

            right.sibling = self.sibling;
            self.sibling = Some(right);

            return NodeRef::LeafRef(right);
        } else {
            let mut value = V::zero();
            update_fn(&mut value);
            self.keyvals.insert(place, (key, value));
            return NodeRef::None;
        }
    }
}
