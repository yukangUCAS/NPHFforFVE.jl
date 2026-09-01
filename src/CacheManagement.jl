"""
    cache_info()

Return entry counts and standalone `Base.summarysize` estimates for the
process-wide geometry/projection caches.  Sizes may overlap because cached
objects can be shared and therefore must not be summed as resident memory.
"""
function cache_info()
    caches = (
        momentum_representatives=_MOM_REP_CACHE,
        helicity_representatives=_HEL_REP_CACHE,
        subspace_states=_SUBSPACE_STATE_CACHE,
        wigner_D=_D_CACHE,
        rotation_vectors=_ROT_VEC_CACHE,
        spin_projections=_SPIN_PROJ_CACHE,
        spin_value_tables=_M_VALS_CACHE,
        wigner_angles=_WIGNER_ANGLE_CACHE,
        projections=_PROJ_CACHE,
        projection_lists=_PROJ_LIST_CACHE,
        projection_orbits=_PROJ_ORBIT_CACHE,
    )
    return NamedTuple{keys(caches)}(
        map(cache -> (entries=length(cache), bytes=Base.summarysize(cache)),
            values(caches)))
end

"""
    clear_caches!()

Clear all process-wide geometry, rotation, and projection caches.  Existing
`SystemBasis` and prepared affine objects remain valid because they own their
computed arrays.  Call this between unrelated geometry scans to release cached
references.  Do not call it concurrently with basis/projection construction.
"""
function clear_caches!()
    empty!(_MOM_REP_CACHE)
    empty!(_HEL_REP_CACHE)
    empty!(_SUBSPACE_STATE_CACHE)
    empty!(_D_CACHE)
    empty!(_ROT_VEC_CACHE)
    empty!(_SPIN_PROJ_CACHE)
    empty!(_M_VALS_CACHE)
    _clear_wigner_angle_cache!()
    empty!(_PROJ_CACHE)
    empty!(_PROJ_LIST_CACHE)
    _clear_projection_orbit_cache!()
    return nothing
end
