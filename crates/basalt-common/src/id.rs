//! Monotonic identifiers.

use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

/// A unique, monotonically increasing identifier.
///
/// Backed by [`NonZeroU64`] so that `Option<Id>` is the same size as `Id`;
/// zero is reserved as the "no id" niche. Ids are only ever produced by an
/// [`IdGenerator`], which makes accidental collisions impossible without an
/// explicit `Id::from_raw`.
///
/// ```
/// use basalt_common::{Id, IdGenerator};
///
/// let ids = IdGenerator::new();
/// let a: Id = ids.next_id();
/// let b: Id = ids.next_id();
/// assert!(a < b);
/// assert_eq!(a.get(), 1);
/// assert_eq!(format!("{b}"), "#2");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id(NonZeroU64);

impl Id {
    /// The raw numeric value.
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// Rebuild an `Id` from a previously persisted raw value.
    ///
    /// Returns `None` for `0`. This is the only way to make an `Id` without a
    /// generator and exists so that ids can survive a round trip through a
    /// segment footer.
    pub const fn from_raw(raw: u64) -> Option<Id> {
        match NonZeroU64::new(raw) {
            Some(v) => Some(Id(v)),
            None => None,
        }
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Thread-safe source of monotonically increasing [`Id`]s.
///
/// The counter uses [`Ordering::Relaxed`]. That is sufficient and deliberate:
/// the only guarantee callers need is that no two `next_id()` calls return the
/// same value, which `fetch_add` provides on its own. We are *not* using the id
/// to publish other memory, so acquire/release would be paying for a fence that
/// nothing observes.
///
/// Allocation-free and lock-free; `next_id` is a single atomic RMW.
#[derive(Debug)]
pub struct IdGenerator {
    next: AtomicU64,
}

impl IdGenerator {
    /// A generator whose first id is `1`.
    pub const fn new() -> Self {
        IdGenerator {
            next: AtomicU64::new(1),
        }
    }

    /// A generator whose first id is `start`, clamped to at least `1`.
    ///
    /// Used when resuming from persisted state: pass `max_seen_id + 1`.
    pub fn starting_at(start: u64) -> Self {
        IdGenerator {
            next: AtomicU64::new(start.max(1)),
        }
    }

    /// Allocate the next identifier.
    ///
    /// # Panics
    ///
    /// Never in practice: it would require 2^64 - 1 calls to exhaust the space.
    /// The saturating fallback below keeps the function total anyway.
    pub fn next_id(&self) -> Id {
        let raw = self.next.fetch_add(1, Ordering::Relaxed);
        match NonZeroU64::new(raw) {
            Some(v) => Id(v),
            // Only reachable after 2^64 allocations wrap the counter back to 0.
            // Returning 1 keeps the API total; uniqueness is already lost at
            // that point and no real deployment gets there.
            None => Id(NonZeroU64::MIN),
        }
    }

    /// The value that the next call to [`IdGenerator::next_id`] will return,
    /// without consuming it. Snapshot only — racy under concurrency by design.
    pub fn peek(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }
}

impl Default for IdGenerator {
    fn default() -> Self {
        IdGenerator::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn ids_are_monotonic_and_unique() {
        let g = IdGenerator::new();
        let ids: Vec<Id> = (0..1000).map(|_| g.next_id()).collect();
        for w in ids.windows(2) {
            assert!(w[0] < w[1]);
        }
        let uniq: std::collections::HashSet<u64> = ids.iter().map(|i| i.get()).collect();
        assert_eq!(uniq.len(), 1000);
        assert_eq!(g.peek(), 1001);
    }

    #[test]
    fn concurrent_ids_are_unique() {
        use std::sync::Arc;
        let g = Arc::new(IdGenerator::new());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let g = Arc::clone(&g);
                std::thread::spawn(move || (0..2000).map(|_| g.next_id().get()).collect::<Vec<_>>())
            })
            .collect();
        let mut all = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        let uniq: std::collections::HashSet<u64> = all.iter().copied().collect();
        assert_eq!(uniq.len(), all.len());
    }

    #[test]
    fn option_id_is_niche_optimised() {
        assert_eq!(std::mem::size_of::<Option<Id>>(), std::mem::size_of::<Id>());
    }

    #[test]
    fn from_raw_rejects_zero() {
        assert!(Id::from_raw(0).is_none());
        assert_eq!(Id::from_raw(42).unwrap().get(), 42);
        assert_eq!(IdGenerator::starting_at(0).next_id().get(), 1);
        assert_eq!(IdGenerator::starting_at(99).next_id().get(), 99);
    }
}
