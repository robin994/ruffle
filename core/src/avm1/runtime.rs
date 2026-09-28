use crate::avm1::function::ExecutionReason;
use crate::avm1::globals::as_broadcaster::BroadcasterFunctions;
use crate::avm1::globals::{as_broadcaster, create_globals};
use crate::avm1::object::stage_object;
use crate::avm1::property_map::PropertyMap;
use crate::avm1::scope::Scope;
use crate::avm1::{Activation, ActivationIdentifier, Error, Object, Value, scope};
use crate::context::UpdateContext;
use crate::display_object::{
    DisplayObject, MovieClip, TDisplayObject, TDisplayObjectContainer, TInteractiveObject,
};
use crate::frame_lifecycle::FramePhase;
use crate::string::{AvmString, StringContext};
use crate::tag_utils::SwfSlice;
use crate::{avm_debug, avm1};
use gc_arena::{Collect, Gc, Mutation};
use std::borrow::Cow;
#[cfg(target_os = "vita")]
use std::collections::{HashMap, HashSet};
#[cfg(target_os = "vita")]
use std::ffi::c_void;
#[cfg(target_os = "vita")]
use std::sync::Arc;
use swf::avm1::read::Reader;
use tracing::instrument;

#[cfg(target_os = "vita")]
unsafe extern "C" {
    fn flashvita_vita_parallel_for(
        count: u32,
        min_grain: u32,
        callback: unsafe extern "C" fn(*mut c_void, u32, u32),
        user: *mut c_void,
    ) -> i32;
}

#[cfg(target_os = "vita")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct VitaBlockKey {
    movie: usize,
    start: usize,
    end: usize,
}

#[cfg(target_os = "vita")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Collect)]
#[collect(require_static)]
struct VitaPushKey {
    movie: usize,
    offset: usize,
}

#[cfg(target_os = "vita")]
#[derive(Clone, Copy, Collect)]
#[collect(no_drop)]
pub(crate) enum VitaPushOperand<'gc> {
    Static(Value<'gc>),
    Register(u8),
    Constant8(u8),
    Constant16(u16),
}

#[cfg(target_os = "vita")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Collect)]
#[collect(require_static)]
struct VitaPushStringKey {
    movie: usize,
    action_offset: usize,
    value_offset: u16,
}

#[cfg(target_os = "vita")]
#[derive(Default)]
struct VitaFastActionCache {
    prepared_blocks: HashSet<VitaBlockKey>,
    push_lengths: HashMap<VitaPushKey, u16>,
    ir_blocks: HashMap<VitaBlockKey, Arc<[VitaIrOp]>>,
}

#[cfg(target_os = "vita")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VitaIrOp {
    /// Absolute byte offset inside the owning SWF movie.
    pub offset: u32,
    pub total_len: u32,
    pub opcode: u8,
    pub push_len: u16,
    pub push_slot: u32,
    pub constant_pool_slot: u32,
    pub branch_target: u32,
}

#[cfg(target_os = "vita")]
pub(crate) const VITA_IR_INVALID_SLOT: u32 = u32::MAX;

#[cfg(target_os = "vita")]
#[derive(Clone, Copy)]
struct VitaPredecodeTask {
    data: *const u8,
    len: usize,
    movie_base: *const u8,
}

#[cfg(target_os = "vita")]
#[derive(Clone, Copy)]
struct VitaPushMeta {
    offset: u32,
    len: u16,
}

#[cfg(target_os = "vita")]
#[derive(Default)]
struct VitaPredecodeOutput {
    pushes: Vec<VitaPushMeta>,
    ops: Vec<VitaIrOp>,
}

#[cfg(target_os = "vita")]
struct VitaPredecodeContext {
    tasks: *const VitaPredecodeTask,
    outputs: *mut VitaPredecodeOutput,
}

#[cfg(target_os = "vita")]
#[inline]
fn vita_read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    let pair = bytes.get(offset..offset.checked_add(2)?)?;
    Some(u16::from_le_bytes([pair[0], pair[1]]))
}

#[cfg(target_os = "vita")]
fn vita_skip_c_string(bytes: &[u8], offset: &mut usize) -> Option<()> {
    let rel = bytes.get(*offset..)?.iter().position(|&b| b == 0)?;
    *offset = (*offset).checked_add(rel + 1)?;
    Some(())
}

#[cfg(target_os = "vita")]
fn vita_action_total_len(bytes: &[u8], offset: usize) -> Option<(u8, usize, Option<u16>)> {
    let opcode = *bytes.get(offset)?;
    if opcode < 0x80 {
        return Some((opcode, 1, None));
    }

    let declared = vita_read_u16(bytes, offset + 1)? as usize;
    let payload_start = offset.checked_add(3)?;
    let declared_end = payload_start.checked_add(declared)?;
    if declared_end > bytes.len() {
        return None;
    }

    let mut extra = 0usize;
    match opcode {
        // DefineFunction: the function body is not included in ActionLength.
        0x9B => {
            let payload = &bytes[payload_start..declared_end];
            let mut pos = 0usize;
            vita_skip_c_string(payload, &mut pos)?;
            let param_count = vita_read_u16(payload, pos)? as usize;
            pos += 2;
            for _ in 0..param_count {
                vita_skip_c_string(payload, &mut pos)?;
            }
            extra = vita_read_u16(payload, pos)? as usize;
        }
        // DefineFunction2: same body-length rule with register metadata.
        0x8E => {
            let payload = &bytes[payload_start..declared_end];
            let mut pos = 0usize;
            vita_skip_c_string(payload, &mut pos)?;
            let param_count = vita_read_u16(payload, pos)? as usize;
            pos += 2;
            pos = pos.checked_add(3)?; // register_count + flags
            for _ in 0..param_count {
                pos = pos.checked_add(1)?; // register index
                vita_skip_c_string(payload, &mut pos)?;
            }
            extra = vita_read_u16(payload, pos)? as usize;
        }
        // Try bodies are stored after the declared action payload.
        0x8F if declared >= 7 => {
            let payload = &bytes[payload_start..declared_end];
            extra = vita_read_u16(payload, 1)? as usize
                + vita_read_u16(payload, 3)? as usize
                + vita_read_u16(payload, 5)? as usize;
        }
        // With body is stored after the declared action payload.
        0x94 if declared >= 2 => {
            extra = vita_read_u16(bytes, payload_start)? as usize;
        }
        _ => {}
    }

    let total = 3usize.checked_add(declared)?.checked_add(extra)?;
    if offset.checked_add(total)? > bytes.len() {
        return None;
    }
    let push_len = (opcode == 0x96).then_some(declared as u16);
    Some((opcode, total, push_len))
}

#[cfg(target_os = "vita")]
fn vita_validate_push_payload(payload: &[u8]) -> bool {
    let mut pos = 0usize;
    while pos < payload.len() {
        let Some(&ty) = payload.get(pos) else {
            return false;
        };
        pos += 1;
        match ty {
            0 => {
                let Some(rel) = payload[pos..].iter().position(|&b| b == 0) else {
                    return false;
                };
                pos += rel + 1;
            }
            1 | 7 => pos += 4,
            2 | 3 => {}
            4 | 5 | 8 => pos += 1,
            6 => pos += 8,
            9 => pos += 2,
            _ => {}
        }
        if pos > payload.len() {
            return false;
        }
    }
    true
}

#[cfg(target_os = "vita")]
unsafe extern "C" fn vita_predecode_worker(user: *mut c_void, begin: u32, end: u32) {
    let context = unsafe { &*(user as *const VitaPredecodeContext) };
    for index in begin as usize..end as usize {
        let task = unsafe { *context.tasks.add(index) };
        let output = unsafe { &mut *context.outputs.add(index) };
        let bytes = unsafe { std::slice::from_raw_parts(task.data, task.len) };
        let mut pos = 0usize;
        while pos < bytes.len() {
            let Some((opcode, total, push_len)) = vita_action_total_len(bytes, pos) else {
                break;
            };
            if let Some(len) = push_len {
                let payload_start = pos + 3;
                let payload_end = payload_start + len as usize;
                if vita_validate_push_payload(&bytes[payload_start..payload_end]) {
                    let action_offset = (task.data as usize + pos)
                        .saturating_sub(task.movie_base as usize);
                    output.pushes.push(VitaPushMeta {
                        offset: action_offset as u32,
                        len,
                    });
                }
            }
            let action_offset = (task.data as usize + pos)
                .saturating_sub(task.movie_base as usize);
            output.ops.push(VitaIrOp {
                offset: action_offset as u32,
                total_len: total as u32,
                opcode,
                push_len: push_len.unwrap_or(0),
                push_slot: VITA_IR_INVALID_SLOT,
                constant_pool_slot: VITA_IR_INVALID_SLOT,
                branch_target: if matches!(opcode, 0x99 | 0x9D) && total >= 5 {
                    let delta = i16::from_le_bytes([bytes[pos + 3], bytes[pos + 4]]) as isize;
                    let target = pos as isize + total as isize + delta;
                    if target >= 0 && target <= task.len as isize {
                        (task.data as usize + target as usize)
                            .saturating_sub(task.movie_base as usize) as u32
                    } else {
                        VITA_IR_INVALID_SLOT
                    }
                } else {
                    VITA_IR_INVALID_SLOT
                },
            });
            pos += total;
            if opcode == 0x00 {
                break;
            }
        }
    }
}

/// The global environment.
///
/// Because SWFs v6 and v7+ use different case-sensitivity rules, Flash
/// keeps two environments, one case-sensitive, the other not (for an
/// example, see the `global_swf6_7_8` test).
#[derive(Collect)]
#[collect(no_drop)]
struct GlobalEnv<'gc> {
    /// The global scope (pre-allocated so that it can be reused by fresh `Activation`s).
    global_scope: Gc<'gc, Scope<'gc>>,

    /// System built-ins that we use internally to construct new objects.
    prototypes: avm1::globals::SystemPrototypes<'gc>,

    /// Cached functions for the AsBroadcaster.
    broadcaster_functions: BroadcasterFunctions<'gc>,

    /// The mappings between symbol names and constructors registered
    /// with `Object.registerClass()`. This is either case-sensitive or case-insensitive.
    constructor_registry: PropertyMap<'gc, Object<'gc>>,
}

impl<'gc> GlobalEnv<'gc> {
    fn create(context: &mut StringContext<'gc>) -> Self {
        let (prototypes, globals, broadcaster_functions) = create_globals(context);
        Self {
            global_scope: Gc::new(context.gc(), Scope::from_global_object(globals)),
            prototypes,
            broadcaster_functions,
            constructor_registry: PropertyMap::new(),
        }
    }
}

#[derive(Collect)]
#[collect(no_drop)]
pub struct Avm1<'gc> {
    /// The Flash Player version we're emulating.
    player_version: u8,

    /// The constant pool to use for new activations from code sources that
    /// don't close over the constant pool they were defined with.
    constant_pool: Gc<'gc, Vec<Value<'gc>>>,

    /// The global environment, dependent on the ambient SWF version.
    env_case_sensitive: GlobalEnv<'gc>,
    env_case_insensitive: GlobalEnv<'gc>,

    /// DisplayObject property map.
    display_properties: stage_object::DisplayPropertyMap<'gc>,

    /// The operand stack (shared across functions).
    stack: Vec<Value<'gc>>,

    /// The register slots (also shared across functions).
    /// `ActionDefineFunction2` defined functions do not use these slots.
    registers: [Value<'gc>; 4],

    /// If a serious error has occurred, or a user has requested it, the AVM may be halted.
    /// This will completely prevent any further actions from being executed.
    halted: bool,

    /// The maximum amount of functions that can be called before a `Error::FunctionRecursionLimit`
    /// is raised. This defaults to 256 but can be changed per movie.
    max_recursion_depth: u16,

    /// Whether a Mouse listener has been registered.
    /// Used to prevent scrolling on web.
    has_mouse_listener: bool,

    #[cfg(target_os = "vita")]
    #[collect(require_static)]
    vita_fast_actions: VitaFastActionCache,

    #[cfg(target_os = "vita")]
    vita_push_strings: HashMap<VitaPushStringKey, AvmString<'gc>>,

    #[cfg(target_os = "vita")]
    vita_push_plans: HashMap<VitaPushKey, Vec<VitaPushOperand<'gc>>>,

    #[cfg(target_os = "vita")]
    vita_push_plan_slots: Vec<Option<Vec<VitaPushOperand<'gc>>>>,

    #[cfg(target_os = "vita")]
    vita_constant_pool_slots: Vec<Option<Gc<'gc, Vec<Value<'gc>>>>>,

    /// The list of all movie clips in execution order.
    clip_exec_list: Option<MovieClip<'gc>>,

    /// If getBounds / getRect is called on a MovieClip with invalid bounds and the
    /// target space is identical to the origin space, but the target is not the
    /// MovieClip itself, the call can return either the default invalid rectangle
    /// (all corners have 0x7ffffff twips) or a special invalid bounds rectangle (all
    /// corners have 0x8000000 twips).
    ///
    /// This boolean is used in this situation. If it's true, the special invalid
    /// bounds rectangle is returned instead of the default invalid rectangle.
    ///
    /// This boolean is set to true if getBounds or getRect is called on a MovieClip
    /// with activation SWF version >= 8 or root movie SWF version >= 8. It is an
    /// internal state changing irreversibly. This means that the getBounds result
    /// of a MovieClip can change by calling getBounds on a different MovieClip.
    ///
    /// More examples of this are in the movieclip_invalid_get_bounds_X tests.
    use_new_invalid_bounds_value: bool,

    #[cfg(feature = "avm_debug")]
    pub debug_output: bool,
}

impl<'gc> Avm1<'gc> {
    pub fn new(context: &mut StringContext<'gc>, player_version: u8) -> Self {
        let gc_context = context.gc();

        Self {
            player_version,
            constant_pool: Gc::new(gc_context, vec![]),
            env_case_insensitive: GlobalEnv::create(context),
            env_case_sensitive: GlobalEnv::create(context),
            display_properties: stage_object::DisplayPropertyMap::new(context),
            stack: vec![],
            registers: [Value::Undefined; 4],
            halted: false,
            max_recursion_depth: 255,
            has_mouse_listener: false,
            #[cfg(target_os = "vita")]
            vita_fast_actions: VitaFastActionCache::default(),
            #[cfg(target_os = "vita")]
            vita_push_strings: HashMap::new(),
            #[cfg(target_os = "vita")]
            vita_push_plans: HashMap::new(),
            #[cfg(target_os = "vita")]
            vita_push_plan_slots: Vec::new(),
            #[cfg(target_os = "vita")]
            vita_constant_pool_slots: Vec::new(),
            clip_exec_list: None,

            #[cfg(feature = "avm_debug")]
            debug_output: false,
            use_new_invalid_bounds_value: false,
        }
    }

    #[cfg(target_os = "vita")]
    pub fn vita_predecode_blocks(&mut self, blocks: &[SwfSlice]) -> (usize, usize, bool) {
        let mut pending = Vec::new();
        let mut pending_keys = Vec::new();

        for block in blocks {
            let movie = std::sync::Arc::as_ptr(&block.movie) as usize;
            let key = VitaBlockKey {
                movie,
                start: block.start,
                end: block.end,
            };
            if self.vita_fast_actions.prepared_blocks.contains(&key) {
                continue;
            }

            let data = block.data();
            pending.push(VitaPredecodeTask {
                data: data.as_ptr(),
                len: data.len(),
                movie_base: block.movie.data().as_ptr(),
            });
            pending_keys.push(key);
        }

        if pending.is_empty() {
            return (0, 0, false);
        }

        let mut outputs = pending
            .iter()
            .map(|task| VitaPredecodeOutput {
                // A zero-payload Push still occupies 3 bytes, so this capacity
                // is a hard upper bound and guarantees no worker-side realloc.
                pushes: Vec::with_capacity(task.len / 3 + 1),
                ops: Vec::with_capacity(task.len / 2 + 1),
            })
            .collect::<Vec<_>>();
        let mut predecode_context = VitaPredecodeContext {
            tasks: pending.as_ptr(),
            outputs: outputs.as_mut_ptr(),
        };

        let parallel = if pending.len() >= 3 {
            unsafe {
                flashvita_vita_parallel_for(
                    pending.len() as u32,
                    1,
                    vita_predecode_worker,
                    (&mut predecode_context as *mut VitaPredecodeContext).cast(),
                ) > 0
            }
        } else {
            false
        };

        if !parallel {
            unsafe {
                vita_predecode_worker(
                    (&mut predecode_context as *mut VitaPredecodeContext).cast(),
                    0,
                    pending.len() as u32,
                );
            }
        }

        let mut push_count = 0usize;
        for (index, mut output) in outputs.into_iter().enumerate() {
            let key = pending_keys[index];
            for push in &output.pushes {
                self.vita_fast_actions.push_lengths.insert(
                    VitaPushKey {
                        movie: key.movie,
                        offset: push.offset as usize,
                    },
                    push.len,
                );
                push_count += 1;
            }
            for op in &mut output.ops {
                if op.opcode == 0x96 {
                    op.push_slot = self.vita_push_plan_slots.len() as u32;
                    self.vita_push_plan_slots.push(None);
                } else if op.opcode == 0x88 {
                    op.constant_pool_slot = self.vita_constant_pool_slots.len() as u32;
                    self.vita_constant_pool_slots.push(None);
                }
            }
            self.vita_fast_actions
                .ir_blocks
                .insert(key, Arc::from(output.ops));
            self.vita_fast_actions.prepared_blocks.insert(key);
        }

        (pending.len(), push_count, parallel)
    }

    #[cfg(target_os = "vita")]
    pub(crate) fn vita_ir_block(&mut self, block: &SwfSlice) -> Arc<[VitaIrOp]> {
        let key = VitaBlockKey {
            movie: Arc::as_ptr(&block.movie) as usize,
            start: block.start,
            end: block.end,
        };
        if !self.vita_fast_actions.prepared_blocks.contains(&key) {
            let _ = self.vita_predecode_blocks(std::slice::from_ref(block));
        }
        self.vita_fast_actions
            .ir_blocks
            .get(&key)
            .cloned()
            .unwrap_or_else(|| Arc::from([]))
    }

    #[cfg(target_os = "vita")]
    #[inline]
    pub fn vita_fast_push_len(&self, movie: usize, offset: usize) -> Option<u16> {
        self.vita_fast_actions
            .push_lengths
            .get(&VitaPushKey { movie, offset })
            .copied()
    }

    #[cfg(target_os = "vita")]
    #[inline]
    pub fn vita_record_fast_push_len(&mut self, movie: usize, offset: usize, len: u16) {
        self.vita_fast_actions
            .push_lengths
            .insert(VitaPushKey { movie, offset }, len);
    }

    #[cfg(target_os = "vita")]
    #[inline]
    pub fn vita_cached_push_string(
        &self,
        movie: usize,
        action_offset: usize,
        value_offset: u16,
    ) -> Option<AvmString<'gc>> {
        self.vita_push_strings
            .get(&VitaPushStringKey {
                movie,
                action_offset,
                value_offset,
            })
            .copied()
    }

    #[cfg(target_os = "vita")]
    #[inline]
    pub fn vita_cache_push_string(
        &mut self,
        movie: usize,
        action_offset: usize,
        value_offset: u16,
        string: AvmString<'gc>,
    ) {
        self.vita_push_strings.insert(
            VitaPushStringKey {
                movie,
                action_offset,
                value_offset,
            },
            string,
        );
    }

    #[cfg(target_os = "vita")]
    #[inline]
    pub(crate) fn vita_push_plan(
        &self,
        movie: usize,
        action_offset: usize,
    ) -> Option<smallvec::SmallVec<[VitaPushOperand<'gc>; 8]>> {
        self.vita_push_plans
            .get(&VitaPushKey {
                movie,
                offset: action_offset,
            })
            .map(|plan| smallvec::SmallVec::from_slice(plan))
    }

    #[cfg(target_os = "vita")]
    #[inline]
    pub(crate) fn vita_cache_push_plan(
        &mut self,
        movie: usize,
        action_offset: usize,
        plan: Vec<VitaPushOperand<'gc>>,
    ) {
        self.vita_push_plans.insert(
            VitaPushKey {
                movie,
                offset: action_offset,
            },
            plan,
        );
    }

    #[cfg(target_os = "vita")]
    #[inline]
    pub(crate) fn vita_push_plan_slot(
        &self,
        slot: u32,
    ) -> Option<smallvec::SmallVec<[VitaPushOperand<'gc>; 8]>> {
        self.vita_push_plan_slots
            .get(slot as usize)
            .and_then(Option::as_ref)
            .map(|plan| smallvec::SmallVec::from_slice(plan))
    }

    #[cfg(target_os = "vita")]
    #[inline]
    pub(crate) fn vita_cache_push_plan_slot(&mut self, slot: u32, plan: Vec<VitaPushOperand<'gc>>) {
        if let Some(entry) = self.vita_push_plan_slots.get_mut(slot as usize) {
            *entry = Some(plan);
        }
    }

    #[cfg(target_os = "vita")]
    #[inline]
    pub(crate) fn vita_constant_pool_slot(&self, slot: u32) -> Option<Gc<'gc, Vec<Value<'gc>>>> {
        self.vita_constant_pool_slots
            .get(slot as usize)
            .and_then(|pool| *pool)
    }

    #[cfg(target_os = "vita")]
    #[inline]
    pub(crate) fn vita_cache_constant_pool_slot(
        &mut self,
        slot: u32,
        pool: Gc<'gc, Vec<Value<'gc>>>,
    ) {
        if let Some(entry) = self.vita_constant_pool_slots.get_mut(slot as usize) {
            *entry = Some(pool);
        }
    }

    pub fn load_player_globals(context: &mut UpdateContext<'gc>) {
        avm1::globals::load_playerglobal(context);
    }

    /// Add a stack frame that executes code in timeline scope
    ///
    /// This creates a new frame stack.
    pub fn run_stack_frame_for_action(
        active_clip: DisplayObject<'gc>,
        name: &str,
        code: SwfSlice,
        context: &mut UpdateContext<'gc>,
    ) {
        if context.avm1.halted {
            // We've been told to ignore all future execution.
            return;
        }

        let clip_obj = active_clip.object1_or_bare(context.gc());
        let child_scope = Gc::new(
            context.gc(),
            Scope::new(
                context.avm1.global_scope(active_clip.swf_version()),
                scope::ScopeClass::Target,
                clip_obj,
            ),
        );
        let constant_pool = context.avm1.constant_pool;
        let mut child_activation = Activation::from_action(
            context,
            ActivationIdentifier::root(name),
            active_clip.swf_version(),
            child_scope,
            constant_pool,
            active_clip,
            clip_obj.into(),
            None,
            &[],
        );
        if let Err(e) = child_activation.run_actions(code) {
            Self::handle_error(&mut child_activation, e);
        }
    }

    /// Add a stack frame that executes code in globals scope
    ///
    /// This creates a new frame stack.
    pub fn run_stack_frame_for_globals(code: SwfSlice, context: &mut UpdateContext<'gc>) {
        if context.avm1.halted {
            // We've been told to ignore all future execution.
            return;
        }

        let constant_pool = context.avm1.constant_pool;
        // Let's do this once for swf 6 (case sensitive) and once for swf 7 (case insensitive),
        // as we keep separate _global objects around for those two cases.
        for swf_version in [6, 7] {
            let mut child_activation = Activation::from_action(
                context,
                ActivationIdentifier::root("playerglobal"),
                swf_version,
                context.avm1.global_scope(swf_version),
                constant_pool,
                context.stage.as_displayobject(),
                Value::Null,
                None,
                &[],
            );
            if let Err(e) = child_activation.run_actions(code.clone()) {
                Self::handle_error(&mut child_activation, e);
            }
        }
    }

    /// Add a stack frame that executes code in initializer scope.
    ///
    /// This creates a new frame stack.
    pub fn run_with_stack_frame_for_display_object<'a, F, R>(
        active_clip: DisplayObject<'gc>,
        action_context: &mut UpdateContext<'gc>,
        function: F,
    ) -> R
    where
        for<'b> F: FnOnce(&mut Activation<'b, 'gc>) -> R,
    {
        let clip_obj = active_clip
            .object1()
            .expect("No script object for display object");
        let child_scope = Gc::new(
            action_context.gc(),
            Scope::new(
                action_context.avm1.global_scope(active_clip.swf_version()),
                scope::ScopeClass::Target,
                clip_obj,
            ),
        );
        let constant_pool = action_context.avm1.constant_pool;
        let mut activation = Activation::from_action(
            action_context,
            ActivationIdentifier::root("[Display Object]"),
            active_clip.swf_version(),
            child_scope,
            constant_pool,
            active_clip,
            clip_obj.into(),
            None,
            &[],
        );
        function(&mut activation)
    }

    /// Add a stack frame that executes code in initializer scope.
    ///
    /// This creates a new frame stack.
    pub fn run_stack_frame_for_init_action(
        active_clip: DisplayObject<'gc>,
        code: SwfSlice,
        context: &mut UpdateContext<'gc>,
    ) {
        if context.avm1.halted {
            // We've been told to ignore all future execution.
            return;
        }

        let clip_obj = active_clip.object1_or_bare(context.gc());
        let child_scope = Gc::new(
            context.gc(),
            Scope::new(
                context.avm1.global_scope(active_clip.swf_version()),
                scope::ScopeClass::Target,
                clip_obj,
            ),
        );
        context.avm1.push(Value::Undefined);
        let constant_pool = context.avm1.constant_pool;
        let mut child_activation = Activation::from_action(
            context,
            ActivationIdentifier::root("[Init]"),
            active_clip.swf_version(),
            child_scope,
            constant_pool,
            active_clip,
            clip_obj.into(),
            None,
            &[],
        );
        if let Err(e) = child_activation.run_actions(code) {
            Self::handle_error(&mut child_activation, e);
        }
    }

    /// Add a stack frame that executes code in timeline scope for an object
    /// method, such as an event handler.
    ///
    /// This creates a new frame stack.
    pub fn run_stack_frame_for_method(
        active_clip: DisplayObject<'gc>,
        obj: Object<'gc>,
        name: AvmString<'gc>,
        args: &[Value<'gc>],
        context: &mut UpdateContext<'gc>,
    ) {
        if context.avm1.halted {
            // We've been told to ignore all future execution.
            return;
        }

        let name_utf8 = &name.to_utf8_lossy();
        let mut activation =
            Activation::from_nothing(context, ActivationIdentifier::root(name_utf8), active_clip);

        let _ = obj.call_method(name, args, &mut activation, ExecutionReason::Special);
    }

    pub fn notify_system_listeners(
        active_clip: DisplayObject<'gc>,
        broadcaster_name: AvmString<'gc>,
        method: AvmString<'gc>,
        args: &[Value<'gc>],
        context: &mut UpdateContext<'gc>,
    ) {
        let mut activation = Activation::from_nothing(
            context,
            ActivationIdentifier::root("[System Listeners]"),
            active_clip,
        );

        let broadcaster = activation
            .global_object()
            .get(broadcaster_name, &mut activation)
            .and_then(|v| v.coerce_to_object_or_bare(&mut activation))
            .unwrap();

        let has_listener =
            as_broadcaster::broadcast_internal(broadcaster, args, method, &mut activation)
                .unwrap_or(false);
        drop(activation);

        if &broadcaster_name == b"Mouse" {
            context.avm1.has_mouse_listener = has_listener;
        }
    }

    /// Returns true if the `Mouse` object has a listener registered.
    /// Used to prevent mouse wheel scrolling on web.
    pub fn has_mouse_listener(&self) -> bool {
        self.has_mouse_listener
    }

    /// Halts the AVM, preventing execution of any further actions.
    ///
    /// If the AVM is currently evaluating an action, it will continue until it realizes that it has
    /// been halted. If an immediate stop is required, an Error must be raised inside of the execution.
    ///
    /// This is most often used when serious errors or infinite loops are encountered.
    pub fn halt(&mut self) {
        if !self.halted {
            self.halted = true;
            tracing::error!("No more actions will be executed in this movie.")
        }
    }

    pub fn stack_len(&self) -> usize {
        self.stack.len()
    }

    pub fn truncate_stack(&mut self, len: usize) {
        self.stack.truncate(len);
    }

    /// Resets the operand stack and the global registers.
    ///
    /// AVM1 bytecode may leave the stack unbalanced, or access global registers
    /// without initializing them, so this method should be called after executing
    /// bytecode to clear any left-overs.
    pub fn clear(&mut self) {
        self.stack.clear();
        self.registers = [Value::Undefined; 4];
    }

    pub fn push(&mut self, value: Value<'gc>) {
        avm_debug!(self, "Stack push {}: {value:?}", self.stack.len());
        self.stack.push(value);
    }

    pub fn pop(&mut self) -> Value<'gc> {
        let value = self.stack.pop().unwrap_or_else(|| {
            tracing::warn!("Avm1::pop: Stack underflow");
            Value::Undefined
        });

        avm_debug!(self, "Stack pop {}: {value:?}", self.stack.len());

        value
    }

    #[inline(always)]
    pub fn is_case_sensitive(swf_version: u8) -> bool {
        swf_version >= 7
    }

    /// Obtain a reference to the global scope.
    pub fn global_scope(&self, swf_version: u8) -> Gc<'gc, Scope<'gc>> {
        if Self::is_case_sensitive(swf_version) {
            self.env_case_sensitive.global_scope
        } else {
            self.env_case_insensitive.global_scope
        }
    }

    /// Obtain system built-in prototypes for this instance.
    pub fn prototypes(&self, swf_version: u8) -> &avm1::globals::SystemPrototypes<'gc> {
        if Self::is_case_sensitive(swf_version) {
            &self.env_case_sensitive.prototypes
        } else {
            &self.env_case_insensitive.prototypes
        }
    }

    /// Obtains the constant pool to use for new activations from code sources that
    /// don't close over the constant pool they were defined with.
    pub fn constant_pool(&self) -> Gc<'gc, Vec<Value<'gc>>> {
        self.constant_pool
    }

    /// Sets the constant pool to use for new activations from code sources that
    /// don't close over the constant pool they were defined with.
    pub fn set_constant_pool(&mut self, constant_pool: Gc<'gc, Vec<Value<'gc>>>) {
        self.constant_pool = constant_pool;
    }

    /// DisplayObject property map.
    pub fn display_properties(&self) -> &stage_object::DisplayPropertyMap<'gc> {
        &self.display_properties
    }

    pub fn max_recursion_depth(&self) -> u16 {
        self.max_recursion_depth
    }

    pub fn set_max_recursion_depth(&mut self, max_recursion_depth: u16) {
        self.max_recursion_depth = max_recursion_depth
    }

    pub fn broadcaster_functions(&self, swf_version: u8) -> BroadcasterFunctions<'gc> {
        if Self::is_case_sensitive(swf_version) {
            self.env_case_sensitive.broadcaster_functions
        } else {
            self.env_case_insensitive.broadcaster_functions
        }
    }

    /// The Flash Player version we're emulating.
    pub fn player_version(&self) -> u8 {
        self.player_version
    }

    pub fn get_register(&self, id: usize) -> Option<&Value<'gc>> {
        self.registers.get(id)
    }

    pub fn get_register_mut(&mut self, id: usize) -> Option<&mut Value<'gc>> {
        self.registers.get_mut(id)
    }

    /// Find all display objects with negative depth recursively
    ///
    /// If an object is pending removal due to being removed by a removeObject tag on the previous frame,
    /// while it had an unload event listener attached, avm1 requires that the object is kept around for one extra frame.
    ///
    /// This will be called at the start of each frame, to gather the objects for removal
    fn find_display_objects_pending_removal(
        obj: DisplayObject<'gc>,
        out: &mut Vec<DisplayObject<'gc>>,
    ) {
        if let Some(parent) = obj.as_container() {
            for child in parent.iter_render_list() {
                if child.avm1_pending_removal() {
                    out.push(child);
                }

                Self::find_display_objects_pending_removal(child, out);
            }
        }
    }

    /// Remove all display objects pending removal
    /// See [`find_display_objects_pending_removal`] for details
    fn remove_pending(context: &mut UpdateContext<'gc>) {
        // Storage for objects to remove
        // Have to do this in two passes to avoid borrow-mut while already borrowed
        let mut out = Vec::new();

        // Find objects to remove
        for level in context.stage.iter_render_list() {
            Self::find_display_objects_pending_removal(level, &mut out);
        }

        for &child in &out {
            // Get the parent of this object
            if let Some(parent_container) = child.parent().and_then(|p| p.as_container()) {
                // Remove it
                parent_container.remove_child_directly(context, child);

                // Update pending removal state
                parent_container
                    .raw_container_mut(context.gc())
                    .update_pending_removals();
            } else {
                // TODO Investigate it. This situation seems impossible, yet it happens.
                tracing::warn!(
                    "AVM1 object pending removal doesn't have a parent, object={:?}, pending removal={:?}",
                    child,
                    out
                );
            }
        }
    }

    // Run a single frame.
    #[instrument(level = "debug", skip_all)]
    pub fn run_frame(context: &mut UpdateContext<'gc>) {
        // Remove pending objects
        Self::remove_pending(context);

        // In AVM1, we only ever execute the idle phase, and all the work that
        // would ordinarily be phased is instead run all at once in whatever order
        // the SWF requests it.
        *context.frame_phase = FramePhase::Idle;

        // AVM1 execution order is determined by the global execution list, based on instantiation order.
        let mut prev: Option<MovieClip<'gc>> = None;
        let mut next = context.avm1.clip_exec_list;
        while let Some(clip) = next {
            next = clip.next_avm1_clip();
            if clip.avm1_removed() {
                // Clean up removed clips from this frame or a previous frame.
                if let Some(prev) = prev {
                    prev.set_next_avm1_clip(context.gc(), next);
                } else {
                    context.avm1.clip_exec_list = next;
                }
                clip.set_next_avm1_clip(context.gc(), None);
            } else {
                clip.run_frame_avm1(context);
                prev = Some(clip);
            }
        }

        // Fire "onLoadInit" events and remove completed movie loaders.
        context
            .load_manager
            .movie_clip_on_load(context.action_queue, &context.strings);

        *context.frame_phase = FramePhase::Idle;

        // Looks like the stack is cleared between frames.
        context.avm1.clear();
    }

    /// Adds a movie clip to the execution list.
    ///
    /// This should be called whenever a movie clip is created, and controls the order of
    /// execution for AVM1 movies.
    pub fn add_to_exec_list(&mut self, gc_context: &Mutation<'gc>, clip: MovieClip<'gc>) {
        // Adding while iterating is safe, as this does not modify any active nodes.
        if clip.next_avm1_clip().is_none() {
            clip.set_next_avm1_clip(gc_context, self.clip_exec_list);
            self.clip_exec_list = Some(clip);
        }
    }

    pub fn get_registered_constructor(
        &self,
        swf_version: u8,
        symbol: AvmString<'gc>,
    ) -> Option<Object<'gc>> {
        let is_case_sensitive = Self::is_case_sensitive(swf_version);
        let registry = if is_case_sensitive {
            &self.env_case_sensitive.constructor_registry
        } else {
            &self.env_case_insensitive.constructor_registry
        };
        registry.get(symbol, is_case_sensitive).copied()
    }

    /// Finds the class name associated with a given constructor function.
    /// This is specifically required for AMF0 TypedObject serialization.
    pub fn get_class_name_by_constructor(
        &self,
        swf_version: u8,
        constructor: Object<'gc>,
    ) -> Option<AvmString<'gc>> {
        let is_case_sensitive = Self::is_case_sensitive(swf_version);
        let registry = if is_case_sensitive {
            &self.env_case_sensitive.constructor_registry
        } else {
            &self.env_case_insensitive.constructor_registry
        };
        // Iterate through the PropertyMap to find the matching constructor reference.
        // If multiple aliases point to the same constructor, this will grab the one
        // based on the PropertyMap's iteration order.
        for (alias, registered_constructor) in registry.iter() {
            // We compare by pointer identity because `Object.registerClass` associates
            // a string with a specific Function object reference in memory. We need to
            // verify it is the exact same object, not just an object with equal values.
            if Object::ptr_eq(*registered_constructor, constructor) {
                return Some(alias);
            }
        }
        None
    }

    pub fn register_constructor(
        &mut self,
        swf_version: u8,
        symbol: AvmString<'gc>,
        constructor: Option<Object<'gc>>,
    ) {
        let is_case_sensitive = Self::is_case_sensitive(swf_version);
        let registry = if is_case_sensitive {
            &mut self.env_case_sensitive.constructor_registry
        } else {
            &mut self.env_case_insensitive.constructor_registry
        };
        if let Some(constructor) = constructor {
            if !is_case_sensitive {
                // In case-insensitive mode, PropertyMap updates the value but preserves
                // the original casing of the key. Flash Player overwrites both.
                // We must remove it first to ensure the new casing is stored.
                // This is necessary for AMF0 serialization of TypedObjects.
                registry.remove(symbol, is_case_sensitive);
            }
            registry.insert(symbol, constructor, is_case_sensitive);
        } else {
            registry.remove(symbol, is_case_sensitive);
        }
    }

    /// Returns use_new_invalid_bounds_value.
    pub fn get_use_new_invalid_bounds_value(&self) -> bool {
        self.use_new_invalid_bounds_value
    }

    /// Sets use_new_invalid_bounds_value to true.
    pub fn activate_use_new_invalid_bounds_value(&mut self) {
        self.use_new_invalid_bounds_value = true;
    }

    #[cfg(feature = "avm_debug")]
    #[inline]
    pub fn show_debug_output(&self) -> bool {
        self.debug_output
    }

    #[cfg(not(feature = "avm_debug"))]
    pub const fn show_debug_output(&self) -> bool {
        false
    }

    #[cfg(feature = "avm_debug")]
    pub fn set_show_debug_output(&mut self, visible: bool) {
        self.debug_output = visible;
    }

    #[cfg(not(feature = "avm_debug"))]
    pub const fn set_show_debug_output(&self, _visible: bool) {}

    pub fn handle_error(activation: &mut Activation<'_, 'gc>, error: Error<'gc>) {
        match &error {
            Error::ThrownValue(value) => {
                tracing::warn!("Uncaught AVM1 error: {value:?}");

                let string = if let Ok(message) = value.coerce_to_string(activation) {
                    Cow::Owned(message.to_utf8_lossy().to_string())
                } else {
                    // The only Value variant that can throw an error when being stringified
                    // is Object, so just print "[type Object]".
                    Cow::Borrowed("[type Object]")
                };

                activation
                    .context
                    .avm_warning(&format!("Uncaught exception, {string}"));

                // Continue execution without halting.
                return;
            }
            Error::InvalidBytecode(swf_error) => {
                tracing::error!("{}: {}", error, swf_error);
            }
            _ => {
                tracing::error!("{}", error);
            }
        }
        activation.context.avm1.halt();
    }
}

/// Utility function used by `Avm1::action_wait_for_frame` and
/// `Avm1::action_wait_for_frame_2`.
pub fn skip_actions(reader: &mut Reader<'_>, num_actions_to_skip: u8) {
    for _ in 0..num_actions_to_skip {
        if let Err(e) = reader.read_action() {
            tracing::warn!("Couldn't skip action: {}", e);
        }
    }
}
