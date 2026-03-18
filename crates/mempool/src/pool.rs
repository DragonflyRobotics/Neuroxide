use crate::pointers::Pointer;
use crate::{
    block::BlockTrait, device_allocator::DeviceAllocator, dummyptr::DummyPtr, pointers::DevicePtr,
};
use core::fmt;
use std::collections::BTreeSet;

use crate::{
    block::{Block, BlockRef},
    segment::Segment,
};

pub struct Pool {
    device: Box<dyn DeviceAllocator>,
    segments: Vec<Segment>,
    heap: BTreeSet<BlockRef>,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <BTreeSet<BlockRef> as std::fmt::Debug>::fmt(&self.heap, f)
    }
}

pub trait PoolTrait {
    const BLOCK_SIZE: usize = 4096;
    fn new(device: Box<dyn DeviceAllocator>) -> Self;

    fn malloc(&mut self, size: usize) -> Option<BlockRef>;

    fn free(&mut self, block: &BlockRef);

    fn print(&self);
}

impl PoolTrait for Pool {
    const BLOCK_SIZE: usize = 4096;
    fn new(device: Box<dyn DeviceAllocator>) -> Pool {
        Pool {
            device,
            segments: Vec::new(),
            heap: BTreeSet::new(),
        }
    }

    fn malloc(&mut self, size: usize) -> Option<BlockRef> {
        let dummy_ptr: Box<dyn DevicePtr> = DummyPtr::new(std::ptr::null_mut::<u8>());
        let dummy = Block::new(size, dummy_ptr);
        let suitable_block = self.heap.range(dummy..).next().cloned();
        if let Some(block) = suitable_block {
            self.heap.remove(&block);
            let other_block = block.split(size);
            block.borrow_mut().is_free = false;
            if let Some(ob) = other_block {
                self.heap.insert(ob);
            }
            Some(block.clone())
        } else {
            let res = self.device.allocate(std::cmp::max(size, Self::BLOCK_SIZE));
            if res.is_err() {
                panic!("Device allocation failed");
            }
            let fresh_block = Block::new(std::cmp::max(size, Self::BLOCK_SIZE), res.unwrap());
            self.heap.insert(fresh_block.clone());
            self.segments.push(Segment {
                head: fresh_block.clone(),
            });
            self.malloc(size)
        }
    }

    fn free(&mut self, block: &BlockRef) {
        block.borrow_mut().is_free = true;
        if !block.join_next(&mut self.heap) && !block.join_prev(&mut self.heap) {
            self.heap.insert(block.clone());
        }
        let mut test_block = block.clone();
        while let Some(prev_weak) = test_block.get_prev() {
            if let Some(prev_block) = prev_weak.upgrade() {
                test_block = prev_block;
            } else {
                break;
            }
        }

        if test_block.get_prev().is_none()
            && test_block.get_next().is_none()
            && test_block.borrow().is_free
        {
            self.device.deallocate(test_block.borrow().ptr.clone());
            // println!("Removing segment with head id: {}", test_block.borrow().id);
            self.segments
                .retain(|seg| seg.head.borrow().id != test_block.borrow().id);
            self.heap.remove(&test_block);
        }
    }

    fn print(&self) {
        println!("Pool State:");
        for segment in &self.segments {
            println!("  Segment: {:?}", segment.head.borrow());
            segment.head.print("\t");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dummydevice::DummyDevice;
    #[test]
    fn test_pool_malloc_free() {
        let mut pool = Pool::new(DummyDevice::new());
        let block1 = pool.malloc(1024).expect("Failed to allocate block1");
        let block2 = pool.malloc(2048).expect("Failed to allocate block2");
        pool.print();

        pool.free(&block1);
        pool.print();

        pool.free(&block2);
        pool.print();
    }
}
