use crate::{PlaybackSettings, VelloAsset};
use bevy::{prelude::*, utils::hashbrown::HashMap};

#[derive(Component, Default)]
pub struct AnimationController {
    current_state: &'static str,
    pending_next_state: Option<&'static str>,
    states: HashMap<&'static str, AnimationState>,
}

impl AnimationController {
    pub fn current_state(&self) -> &AnimationState {
        self.states
            .get(self.current_state)
            .unwrap_or_else(|| panic!("state not found: '{}'", self.current_state))
    }

    pub fn transition(&mut self, state: &'static str) {
        self.pending_next_state.replace(state);
    }
}

pub struct AnimationState {
    pub id: &'static str,
    pub asset: Handle<VelloAsset>,
    pub playback_settings: Option<PlaybackSettings>,
    pub transitions: Vec<AnimationTransition>,
}

#[allow(clippy::enum_variant_names)]
pub enum AnimationTransition {
    /// Transitions after a set period of seconds.
    OnAfter {
        state: &'static str,
        secs: f32,
    },
    /// Transition to a different state after all frames complete.
    ///
    /// # Panics
    /// Panics if this state transition was attached to an SVG asset, which isn't supported. Use `OnAfter` instead.
    OnComplete {
        state: &'static str,
    },
    OnMouseEnter {
        state: &'static str,
    },
    OnMouseClick {
        state: &'static str,
    },
    OnMouseLeave {
        state: &'static str,
    },
}

impl AnimationController {
    pub fn new(initial_state: &'static str) -> AnimationController {
        AnimationController {
            current_state: initial_state,
            pending_next_state: Some(initial_state),
            states: HashMap::new(),
        }
    }

    pub fn with_state(mut self, state: AnimationState) -> Self {
        self.states.insert(state.id, state);
        self
    }
}

impl AnimationState {
    pub fn new(id: &'static str) -> Self {
        Self {
            id,
            asset: Default::default(),
            playback_settings: None,
            transitions: vec![],
        }
    }

    pub fn with_asset(mut self, asset: Handle<VelloAsset>) -> Self {
        self.asset = asset;
        self
    }

    pub fn with_playback_settings(mut self, playback_settings: PlaybackSettings) -> Self {
        self.playback_settings.replace(playback_settings);
        self
    }

    pub fn with_transition(mut self, transition: AnimationTransition) -> Self {
        self.transitions.push(transition);
        self
    }
}

pub struct AnimationControllerPlugin;

impl Plugin for AnimationControllerPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(
            Update,
            (systems::run_transitions, systems::set_animation_for_state).chain(),
        );
    }
}

pub mod systems {
    use super::{AnimationController, AnimationTransition};
    use crate::{PlaybackSettings, Vector, VelloAsset};
    use bevy::{prelude::*, utils::Instant};

    pub fn set_animation_for_state(
        mut commands: Commands,
        mut query_sm: Query<(Entity, &mut AnimationController, &mut Handle<VelloAsset>)>,
        mut assets: ResMut<Assets<VelloAsset>>,
    ) {
        for (entity, mut controller, mut cur_handle) in query_sm.iter_mut() {
            let Some(next_state) = controller.pending_next_state.take() else {
                continue;
            };
            let target_state = controller
                .states
                .get(&next_state)
                .unwrap_or_else(|| panic!("state not found: '{}'", next_state));

            info!("animation controller transitioning to={next_state}");
            let target_asset = assets.get_mut(target_state.asset.id()).unwrap();
            match &mut target_asset.data {
                Vector::Svg {
                    playback_started, ..
                }
                | Vector::Lottie {
                    playback_started, ..
                } => {
                    *playback_started = Instant::now();
                }
            };
            *cur_handle = target_state.asset.clone();
            commands.entity(entity).remove::<PlaybackSettings>();
            if let Some(playback_settings) = &target_state.playback_settings {
                commands.entity(entity).insert(playback_settings.clone());
            }
            controller.current_state = next_state;
        }
    }

    pub fn run_transitions(
        mut query_sm: Query<(
            &mut AnimationController,
            &GlobalTransform,
            Option<&PlaybackSettings>,
            &mut Handle<VelloAsset>,
        )>,
        mut assets: ResMut<Assets<VelloAsset>>,

        // For input events
        windows: Query<&Window>,
        query_view: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
        buttons: Res<Input<MouseButton>>,
        mut hovered: Local<bool>,
    ) {
        let window = windows.single();
        let (camera, view) = query_view.single();

        let pointer_pos = window
            .cursor_position()
            .and_then(|cursor| camera.viewport_to_world(view, cursor))
            .map(|ray| ray.origin.truncate());

        for (mut controller, gtransform, playback_settings, current_asset_handle) in
            query_sm.iter_mut()
        {
            let current_state_name = controller.current_state.to_owned();
            let current_asset_id = current_asset_handle.id();

            let current_state = controller.current_state();
            let current_asset = assets
                .get_mut(current_asset_id)
                .unwrap_or_else(|| panic!("asset not found for state: '{current_state_name}'"));

            let is_inside = {
                match pointer_pos {
                    Some(pointer_pos) => {
                        let local_transform = current_asset
                            .local_transform_center
                            .compute_matrix()
                            .inverse();
                        let transform = gtransform.compute_matrix() * local_transform;
                        let mouse_local = transform
                            .inverse()
                            .transform_point3(pointer_pos.extend(0.0));
                        mouse_local.x <= current_asset.width
                            && mouse_local.x >= 0.0
                            && mouse_local.y >= -current_asset.height
                            && mouse_local.y <= 0.0
                    }
                    None => false,
                }
            };

            for transition in current_state.transitions.iter() {
                match transition {
                    AnimationTransition::OnAfter { state, secs } => {
                        let started = match current_asset.data {
                            Vector::Svg {
                                playback_started, ..
                            }
                            | Vector::Lottie {
                                playback_started, ..
                            } => playback_started,
                        };
                        let elapsed_dt = started.elapsed().as_secs_f32();
                        if elapsed_dt > *secs {
                            controller.pending_next_state = Some(state);
                            break;
                        }
                    }
                    AnimationTransition::OnComplete { state } => {
                        match &current_asset.data {
                            crate::Vector::Svg {..} => panic!("invalid state: '{}', `OnComplete` is only valid for Lottie files. Use `OnAfter` for SVG.", current_state.id),
                            crate::Vector::Lottie {
                                original,
                                playback_started, ..
                            } => {
                                let mut elapsed_dt=
                                playback_started.elapsed().as_secs_f32();
                                if let Some(playback_settings) = playback_settings {
                                    elapsed_dt *= playback_settings.speed;
                                }
                                let complete_dt = (original.frames.end - original.frames.start).abs() / original.frame_rate;
                                if elapsed_dt > complete_dt {
                                    controller.pending_next_state = Some(state);
                                    break;
                                }
                            },
                        };
                    }
                    AnimationTransition::OnMouseEnter { state } => {
                        if is_inside {
                            controller.pending_next_state = Some(state);
                            break;
                        }
                    }
                    AnimationTransition::OnMouseClick { state } => {
                        if is_inside && buttons.just_pressed(MouseButton::Left) {
                            controller.pending_next_state = Some(state);
                            break;
                        }
                    }
                    AnimationTransition::OnMouseLeave { state } => {
                        if *hovered && !is_inside {
                            controller.pending_next_state = Some(state);
                            *hovered = false;
                            break;
                        } else if is_inside {
                            *hovered = true;
                        }
                    }
                }
            }
        }
    }
}
