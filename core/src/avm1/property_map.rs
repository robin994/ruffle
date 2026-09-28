//! The map of property names to values used by the ActionScript VM.
//! This allows for dynamically choosing case-sensitivity at runtime,
//! because SWFv6 and below is case-insensitive. This also maintains
//! the insertion order of properties, which is necessary for accurate
//! enumeration order.

use crate::string::{AvmAtom, AvmString, WStr, utils as string_utils};
use fnv::{FnvBuildHasher, FnvHashMap};
use gc_arena::Collect;
use indexmap::{Equivalent, IndexMap};
use std::hash::{Hash, Hasher};

type FnvIndexMap<K, V> = IndexMap<K, V, FnvBuildHasher>;

/// A map from property names to values.
#[derive(Clone, Debug, Collect)]
#[collect(no_drop)]
pub struct PropertyMap<'gc, V> {
    ordered: FnvIndexMap<PropertyName<'gc>, V>,
    atom_index: FnvHashMap<AvmAtom<'gc>, usize>,
}

impl<V> Default for PropertyMap<'_, V> {
    fn default() -> Self {
        Self {
            ordered: FnvIndexMap::default(),
            atom_index: FnvHashMap::default(),
        }
    }
}

impl<'gc, V> PropertyMap<'gc, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn contains_key<T: AsRef<WStr>>(&self, key: T, case_sensitive: bool) -> bool {
        if case_sensitive {
            self.ordered.contains_key(&CaseSensitive(key.as_ref()))
        } else {
            self.ordered.contains_key(&CaseInsensitive(key.as_ref()))
        }
    }

    #[inline]
    pub fn contains_avm_string(&self, key: AvmString<'gc>, case_sensitive: bool) -> bool {
        if case_sensitive
            && let Some(atom) = key.as_interned()
            && self.atom_index.contains_key(&atom)
        {
            return true;
        }
        self.contains_key(key, case_sensitive)
    }

    #[inline]
    pub fn contains_interned_key(&self, key: AvmAtom<'gc>) -> bool {
        self.atom_index.contains_key(&key)
    }

    pub fn entry<'a>(&'a mut self, key: AvmString<'gc>, case_sensitive: bool) -> Entry<'gc, 'a, V> {
        if case_sensitive {
            let atom = key.as_interned();
            let index = atom
                .and_then(|atom| self.atom_index.get(&atom).copied())
                .or_else(|| self.ordered.get_index_of(&CaseSensitive(key.as_ref())));
            if let (Some(atom), Some(index)) = (atom, index) {
                self.atom_index.insert(atom, index);
            }
            match index {
                Some(index) => Entry::Occupied(OccupiedEntry {
                    map: &mut self.ordered,
                    atom_index: &mut self.atom_index,
                    index,
                }),
                None => Entry::Vacant(VacantEntry {
                    map: &mut self.ordered,
                    atom_index: &mut self.atom_index,
                    key,
                }),
            }
        } else {
            match self.ordered.get_index_of(&CaseInsensitive(key.as_ref())) {
                Some(index) => Entry::Occupied(OccupiedEntry {
                    map: &mut self.ordered,
                    atom_index: &mut self.atom_index,
                    index,
                }),
                None => Entry::Vacant(VacantEntry {
                    map: &mut self.ordered,
                    atom_index: &mut self.atom_index,
                    key,
                }),
            }
        }
    }

    /// Gets the value for the specified property.
    pub fn get<T: AsRef<WStr>>(&self, key: T, case_sensitive: bool) -> Option<&V> {
        if case_sensitive {
            self.ordered.get(&CaseSensitive(key.as_ref()))
        } else {
            self.ordered.get(&CaseInsensitive(key.as_ref()))
        }
    }

    #[inline]
    pub fn get_avm_string(&self, key: AvmString<'gc>, case_sensitive: bool) -> Option<&V> {
        if case_sensitive
            && let Some(atom) = key.as_interned()
            && let Some(value) = self.get_interned(atom)
        {
            return Some(value);
        }
        self.get(key, case_sensitive)
    }

    #[inline]
    pub fn get_avm_string_mut(
        &mut self,
        key: AvmString<'gc>,
        case_sensitive: bool,
    ) -> Option<&mut V> {
        if case_sensitive
            && let Some(atom) = key.as_interned()
            && self.atom_index.contains_key(&atom)
        {
            return self.get_interned_mut(atom);
        }
        self.get_mut(key, case_sensitive)
    }

    #[inline]
    pub fn get_interned(&self, key: AvmAtom<'gc>) -> Option<&V> {
        self.atom_index
            .get(&key)
            .and_then(|&index| self.ordered.get_index(index).map(|(_, value)| value))
    }

    /// Gets a mutable reference to the value for the specified property.
    pub fn get_mut<T: AsRef<WStr>>(&mut self, key: T, case_sensitive: bool) -> Option<&mut V> {
        if case_sensitive {
            self.ordered.get_mut(&CaseSensitive(key.as_ref()))
        } else {
            self.ordered.get_mut(&CaseInsensitive(key.as_ref()))
        }
    }

    #[inline]
    pub fn get_interned_mut(&mut self, key: AvmAtom<'gc>) -> Option<&mut V> {
        let index = *self.atom_index.get(&key)?;
        self.ordered.get_index_mut(index).map(|(_, value)| value)
    }

    /// Gets a value by index, based on insertion order.
    pub fn get_index(&self, index: usize) -> Option<&V> {
        self.ordered.get_index(index).map(|(_, v)| v)
    }

    pub fn insert(&mut self, key: AvmString<'gc>, value: V, case_sensitive: bool) -> Option<V> {
        match self.entry(key, case_sensitive) {
            Entry::Occupied(entry) => Some(entry.insert(value)),
            Entry::Vacant(entry) => {
                entry.insert(value);
                None
            }
        }
    }

    /// Returns the value tuples in Flash's iteration order (most recently added first).
    pub fn iter(&self) -> impl Iterator<Item = (AvmString<'gc>, &V)> {
        self.ordered.iter().rev().map(|(k, v)| (k.0, v))
    }

    /// Returns the key-value tuples in Flash's iteration order (most recently added first).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (AvmString<'gc>, &mut V)> {
        self.ordered.iter_mut().rev().map(|(k, v)| (k.0, v))
    }

    pub fn remove<T: AsRef<WStr>>(&mut self, key: T, case_sensitive: bool) -> Option<V> {
        // Note that we must use shift_remove to maintain order in case this object is enumerated.
        let removed = if case_sensitive {
            self.ordered.shift_remove(&CaseSensitive(key.as_ref()))
        } else {
            self.ordered.shift_remove(&CaseInsensitive(key.as_ref()))
        };
        if removed.is_some() {
            self.rebuild_atom_index();
        }
        removed
    }

    fn rebuild_atom_index(&mut self) {
        self.atom_index.clear();
        for (index, (name, _)) in self.ordered.iter().enumerate() {
            if let Some(atom) = name.0.as_interned() {
                self.atom_index.insert(atom, index);
            }
        }
    }
}

pub enum Entry<'gc, 'a, V> {
    Occupied(OccupiedEntry<'gc, 'a, V>),
    Vacant(VacantEntry<'gc, 'a, V>),
}

pub struct OccupiedEntry<'gc, 'a, V> {
    map: &'a mut FnvIndexMap<PropertyName<'gc>, V>,
    atom_index: &'a mut FnvHashMap<AvmAtom<'gc>, usize>,
    index: usize,
}

impl<'gc, V> OccupiedEntry<'gc, '_, V> {
    pub fn remove_entry(&mut self) -> (AvmString<'gc>, V) {
        let (k, v) = self.map.shift_remove_index(self.index).unwrap();
        self.atom_index.clear();
        for (index, (name, _)) in self.map.iter().enumerate() {
            if let Some(atom) = name.0.as_interned() {
                self.atom_index.insert(atom, index);
            }
        }
        (k.0, v)
    }

    pub fn get(&self) -> &V {
        self.map.get_index(self.index).unwrap().1
    }

    pub fn get_mut(&mut self) -> &mut V {
        self.map.get_index_mut(self.index).unwrap().1
    }

    pub fn insert(self, value: V) -> V {
        std::mem::replace(self.map.get_index_mut(self.index).unwrap().1, value)
    }
}

pub struct VacantEntry<'gc, 'a, V> {
    map: &'a mut FnvIndexMap<PropertyName<'gc>, V>,
    atom_index: &'a mut FnvHashMap<AvmAtom<'gc>, usize>,
    key: AvmString<'gc>,
}

impl<V> VacantEntry<'_, '_, V> {
    pub fn insert(self, value: V) {
        let index = self.map.len();
        if let Some(atom) = self.key.as_interned() {
            self.atom_index.insert(atom, index);
        }
        self.map.insert(PropertyName(self.key), value);
    }
}

/// Wraps a str-like type, causing the hash map to use a case insensitive hash and equality.
struct CaseInsensitive<T>(T);

impl Hash for CaseInsensitive<&WStr> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        swf_hash_string_ignore_case(self.0, state);
    }
}

impl<'gc> Equivalent<PropertyName<'gc>> for CaseInsensitive<&WStr> {
    fn equivalent(&self, key: &PropertyName<'gc>) -> bool {
        key.0.eq_ignore_case(self.0)
    }
}

/// Wraps an str-like type, causing the property map to use a case insensitive hash lookup,
/// but case sensitive equality.
struct CaseSensitive<T>(T);

impl Hash for CaseSensitive<&WStr> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        swf_hash_string_ignore_case(self.0, state);
    }
}

impl<'gc> Equivalent<PropertyName<'gc>> for CaseSensitive<&WStr> {
    fn equivalent(&self, key: &PropertyName<'gc>) -> bool {
        key.0 == self.0
    }
}

/// The property keys stored in the property map.
/// This uses a case insensitive hash to ensure that properties can be found in
/// SWFv6, which is case insensitive. The equality check is handled by the `Equivalent`
/// impls above, which allow it to be either case-sensitive or insensitive.
/// Note that the property of if key1 == key2 -> hash(key1) == hash(key2) still holds.
#[derive(Debug, Clone, PartialEq, Eq, Collect)]
#[collect(no_drop)]
struct PropertyName<'gc>(AvmString<'gc>);

impl Hash for PropertyName<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        swf_hash_string_ignore_case(self.0.as_ref(), state);
    }
}

fn swf_hash_string_ignore_case<H: Hasher>(s: &WStr, state: &mut H) {
    s.iter()
        .for_each(|c| string_utils::swf_to_lowercase(c).hash(state));
    state.write_u8(0xff);
}
