use bevy_math::{DMat2, DVec2, DVec3, Mat4, Vec2, Vec3, Vec4Swizzles};

fn into_orthonormal_basis(v: (DVec3, DVec3)) -> (DVec3, DVec3) {
    let basis_0 = v.0.normalize_or_zero();
    // Remove any component of v.1 that lies on basis_0.
    // If the rows are colinear, this axis will end up zero.
    let orthogonal_v1 = v.1 - basis_0 * basis_0.dot(v.1);
    let basis_1 = orthogonal_v1.normalize_or_zero();
    (basis_0, basis_1)
}

// This is equivilent to multiplying by a 3x2 row-major matrix.
fn project_into_basis(basis: (DVec3, DVec3), v: DVec3) -> DVec2 {
    DVec2::new(basis.0.dot(v), basis.1.dot(v))
}

// An outter product is v * transpose(v).
fn outter_product(v: DVec2) -> DMat2 {
    DMat2::from_cols_array(&[v.x * v.x, v.x * v.y, v.y * v.x, v.y * v.y])
}

fn solve_quadratic_equation(a: f64, b: f64, c: f64) -> (f64, f64) {
    let part = b * b - 4.0 * a * c;

    // There should always be solutions for quadratic equations we generate.
    // However - numerical impercision can sometimes result in a slightly negative value.
    let sqrt_part = part.max(0.0).sqrt();

    let v1 = (-b + sqrt_part) / (2.0 * a);
    let v2 = (-b - sqrt_part) / (2.0 * a);

    (v1, v2)
}

// Find lambda where det(m - I * lambda) = 0
fn eigenvalues(m: DMat2) -> (f64, f64) {
    let a = 1.0;
    let b = -(m.x_axis.x + m.y_axis.y);
    let c = m.x_axis.x * m.y_axis.y - m.x_axis.y * m.y_axis.x;
    solve_quadratic_equation(a, b, c)
}

// Find the maximum value of length(m * v) when length(v) == 1
// This is equivient to maximizing the squared scale: dot(m.0, v)**2 + dot(m.1, v)**2 where dot(v, v) ==1.
// This can be rewritten as maximizing: dot(v, (outer_product(m.0) + outer_product(m.1)) * v) under the same constraint.
// This is equivilent to finding the maximum eigenvalue of (outer_product(m.0) + outer_product(m.1)).
fn maximum_scale(m: (DVec2, DVec2)) -> f64 {
    // Without double precision math, this function can have numerical error on the order of 0.001.
    // This is a bit high since framebuffers dimensions are thousands of pixels.
    let m_scale_squared = outter_product(m.0) + outter_product(m.1);
    let e = eigenvalues(m_scale_squared);
    e.0.max(e.1).sqrt()
}

fn make_scale_vector(to_clip_scale: Vec3, viewport_scale: f32) -> DVec3 {
    // Note: the division by two is required because the X and Y axes of clip space have length 2.
    to_clip_scale.as_dvec3() * (viewport_scale as f64) * 0.5
}

/// Determine the maximum amount that space will be streched when projecting onto the viewport.
/// If a camera-aligned image is projected onto the viewport, this function tells you how many pixels are needed per unit of the original space to match the resolution of the viewport.
pub fn maximum_scale_in_viewport(to_clip: Mat4, viewport: Vec2) -> f32 {
    // We want to calculate the maximum amount space can be stretched when the original space is mapped into viewport space.
    // You can imagine a ball of unit-length direction vectors being projected into viewport space.
    // The projection can non-uniformaly stretch and squash the direction vectors.
    // We are trying to determine the length of the longest projected direction vector that is also coplanar with the viewport.
    // Because we are dealing with direction vectors and not positions, we can ignore the rightmost column of the projection matrix.
    // However: analizing how the direction vectors stretch assumes projection is a linear operation, but it is not. Division by W affects the scale.
    // You will need to divide the result by W at the specific point you are considering for this scale to be valid.
    // If the direction that W changes is not orthogonal to the viewport axes, then this calculation will be incorrect, even when dividing by W.
    // Doing the calculation correctly, however, would require solving a more complicated non-linear problem. A non-orthogonal W is very uncommon, however, it is simpler to accept the inaccuracy.

    // These two vectors indicate the the X and Y axes of the viewport, but embedded in the original space.
    // The magnitude of these vectors is the scaling factor between the original space and between each axis of the viewport.
    let viewport_scale = (
        make_scale_vector(to_clip.row(0).xyz(), viewport.x),
        make_scale_vector(to_clip.row(1).xyz(), viewport.y),
    );

    // We create two orthonormal basis vectors that define a 2d subspace coplanar with the viewport scale vectors.
    // The direction of maximum scale will exist in this subspace. (This may not be correct if W is not constant in this subspace.)
    // If the viewport scale vectors are colinear, one of the basis vectors will be zero.
    // One or both of the viewport scale vectors may also be zero, resulting in one or two zero-length basis vectors.
    // A zero-length basis vector corresponds with there being a zero-length scale in one direction.
    // The subsequent math works even if there's a missing dimension, so we don't bother making sure it's a complete basis.
    let subspace_basis = into_orthonormal_basis(viewport_scale);

    // We can now project the viewport scale vectors onto the basis vectors.
    // Because the original scale vectors are co-planar with the basis, the resuling vectors have the same scaling factor as the originals but one less dimension.
    // Note that if the viewport scale vectors are orthogonal, the subspace vectors will be: (|viewport_scale_x|, 0) and (0, |viewport_scale_y|).
    let viewport_scale_in_subspace = (
        project_into_basis(subspace_basis, viewport_scale.0),
        project_into_basis(subspace_basis, viewport_scale.1),
    );

    // Compute the maximum scale in the subspace.
    // Reducing the problem to a 2d subspace means we only need to solve a quadratic equation rather than a cubic equation.
    maximum_scale(viewport_scale_in_subspace) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::camera::{CameraProjection, OrthographicProjection};
    use bevy::math::Quat;

    macro_rules! assert_approx_eq {
        ($x:expr, $y:expr, $d:expr) => {{
            let (xval, yval, dval) = ($x, $y, $d);
            assert!(!f32::is_nan(xval));
            assert!(!f32::is_nan(yval));
            // The tolerance scales with the magnitude of number we're dealing with.
            if (xval - yval).abs() >= xval.abs().min(yval.abs()) * dval {
                panic!(
                    "assertion failed: `(left !== right)` \
                 (left: `{}`, right: `{}`, tolerance: `{}`)",
                    xval, yval, dval
                );
            }
        }};
    }

    #[test]
    fn test_identity_scale() {
        // Identity transform into clip space means that [-1.0, 1.0] - two units of the original space - will appear in the viewport.
        // The scale is therefore 0.5.
        assert_eq!(maximum_scale_in_viewport(Mat4::IDENTITY, Vec2::ONE), 0.5);
    }

    // Different aspect ratios to try and induce numerical error.
    fn viewport_sizes() -> Vec<Vec2> {
        [
            // 5:4
            Vec2::new(600.0, 480.0),
            // 4:3
            Vec2::new(640.0, 480.0),
            // 3:2
            Vec2::new(720.0, 480.0),
            // 16:10
            Vec2::new(768.0, 480.0),
            // 15:9
            Vec2::new(800.0, 480.0),
            // 16:9
            Vec2::new(1920.0, 1080.0),
            // 17:9
            Vec2::new(2048.0, 1080.0),
            // 18:9
            Vec2::new(2160.0, 1080.0),
            // 21:9
            Vec2::new(2560.0, 1080.0),
            // 32:9
            Vec2::new(3840.0, 1080.0),
        ]
        .into()
    }

    // These should _not_ matter, but check them to double check this.
    fn depth_ranges() -> Vec<(f32, f32)> {
        [(-1000.0, 1000.0), (0.0, 1000.0), (-1.0, 1.0)].into()
    }

    // f64 because the orthographic camera scale is the inverse of the scale factor we are computing.
    // This means we need to take an inverse when checking equality.
    // f64 reduces numerical imprecision coming from the testing itself.
    fn ortho_scales() -> Vec<f64> {
        [1.0, 3.0, 7.0, 1.0 / 3.0, 1.0 / 7.0].into()
    }

    fn camera_rotations() -> Vec<Quat> {
        [
            Quat::IDENTITY,
            Quat::from_scaled_axis(Vec3::new(1.0, 1.0, 1.0)),
            Quat::from_scaled_axis(Vec3::new(0.5, -2.0, 3.0)),
            Quat::from_scaled_axis(Vec3::new(0.7, 0.1, -3.0)),
            Quat::from_scaled_axis(Vec3::new(-0.1, 0.5, 0.3)),
        ]
        .into()
    }

    fn camera_scales() -> Vec<Vec3> {
        [
            Vec3::ONE,
            Vec3::new(3.0, 2.0, 10000.0),
            Vec3::new(0.5, -7.0, 0.001),
        ]
        .into()
    }

    #[test]
    fn test_orthographic_scale() {
        for viewport in viewport_sizes() {
            for (near, far) in depth_ranges() {
                for ortho_scale in ortho_scales() {
                    for camera_scale in camera_scales() {
                        for rot in camera_rotations() {
                            let mut ortho = OrthographicProjection {
                                near,
                                far,
                                scale: ortho_scale as f32,
                                ..OrthographicProjection::default_2d()
                            };
                            ortho.update(viewport.x, viewport.y);

                            // The scaling is done last so we don't need to worry about the Z scale rotating into the viewport.
                            let world_to_camera =
                                Mat4::from_scale(camera_scale) * Mat4::from_quat(rot);
                            let camera_to_clip = ortho.get_clip_from_view();
                            // TODO: why is this backwards?
                            let world_to_clip = camera_to_clip * world_to_camera;

                            assert_approx_eq!(
                                maximum_scale_in_viewport(world_to_clip, viewport),
                                (camera_scale.x.abs().max(camera_scale.y.abs()) as f64
                                    / ortho_scale) as f32,
                                1e-6
                            );
                        }
                    }
                }
            }
        }
    }
}
