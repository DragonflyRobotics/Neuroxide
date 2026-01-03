use std::{
    cell::RefCell,
    collections::BTreeSet,
    fmt,
    rc::{Rc, Weak},
};

use uuid::Uuid;

use crate::pointers::DevicePtr;

pub type BlockRef = Rc<RefCell<Block>>;
pub type WeakBlockRef = Weak<RefCell<Block>>;

pub trait BlockTrait {
    fn split(&self, split_size: usize) -> Option<BlockRef>;
    fn join_next(&self, heap: &mut BTreeSet<BlockRef>) -> bool;
    fn join_prev(&self, heap: &mut BTreeSet<BlockRef>) -> bool;
    fn get_next(&self) -> Option<BlockRef>;
    fn get_prev(&self) -> Option<WeakBlockRef>;
    fn print(&self, prefix: &str);
}

pub struct Block {
    size: usize,
    pub ptr: Box<dyn DevicePtr>,
    pub id: Uuid,
    prev: Option<WeakBlockRef>,
    next: Option<BlockRef>,
    pub is_free: bool,
}

impl std::cmp::PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self.id == other.id
    }
}

impl std::cmp::Eq for Block {}

impl std::cmp::PartialOrd for Block {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl std::cmp::Ord for Block {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.size.cmp(&other.size).then(self.id.cmp(&other.id))
    }
}

impl Block {
    pub fn new(size: usize, ptr: Box<dyn DevicePtr>) -> BlockRef {
        Rc::new(RefCell::new(Block {
            size,
            ptr,
            id: Uuid::new_v4(),
            prev: None,
            next: None,
            is_free: true,
        }))
    }
}

impl fmt::Debug for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Block {{ size: {}, is_free: {}, id: {} }}",
            self.size, self.is_free, self.id
        )
    }
}

impl BlockTrait for BlockRef {
    fn split(self: &BlockRef, split_size: usize) -> Option<BlockRef> {
        if split_size >= self.borrow().size || !self.borrow().is_free {
            return None;
        }

        let remaining_size = self.borrow().size - split_size;
        self.borrow_mut().size = split_size;

        let new_block = Rc::new(RefCell::new(Block {
            size: remaining_size,
            ptr: self.borrow().ptr.clone() + split_size,
            id: Uuid::new_v4(),
            prev: Some(Rc::downgrade(self)),
            next: self.borrow().next.clone(),
            is_free: true,
        }));

        if let Some(next_block) = &self.borrow().next {
            next_block.borrow_mut().prev = Some(Rc::downgrade(&new_block));
        }

        self.borrow_mut().next = Some(new_block.clone());
        Some(new_block)
    }
    fn join_next(&self, heap: &mut BTreeSet<BlockRef>) -> bool {
        if let Some(next_block) = &self.get_next()
            && next_block.borrow().is_free
        {
            heap.remove(next_block);
            self.borrow_mut().size += next_block.borrow().size;
            self.borrow_mut().next = next_block.borrow().next.clone();
            if let Some(next_next) = &next_block.borrow().next {
                next_next.borrow_mut().prev = Some(Rc::downgrade(self));
            }
            heap.insert(self.clone());
            return true;
        }
        false
    }
    fn join_prev(&self, heap: &mut BTreeSet<BlockRef>) -> bool {
        if let Some(prev_weak) = &self.get_prev()
            && let Some(prev_block) = prev_weak.upgrade()
            && prev_block.borrow().is_free
        {
            heap.remove(self);
            heap.remove(&prev_block);
            prev_block.borrow_mut().size += self.borrow().size;
            prev_block.borrow_mut().next = self.borrow().next.clone();
            if let Some(next_block) = &self.borrow().next {
                next_block.borrow_mut().prev = Some(Rc::downgrade(&prev_block));
            }
            heap.insert(prev_block.clone());
            println!("{:?}", prev_block.get_prev());
            return true;
        }
        false
    }
    fn get_next(&self) -> Option<BlockRef> {
        self.borrow().next.clone()
    }
    fn get_prev(&self) -> Option<WeakBlockRef> {
        self.borrow().prev.clone()
    }
    fn print(&self, prefix: &str) {
        println!("{}{:?}", prefix, self.borrow());
        if let Some(next_block) = &self.borrow().next {
            next_block.print(prefix);
        }
    }
}
