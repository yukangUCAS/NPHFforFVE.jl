using NPHFforFVE

include(joinpath(@__DIR__, "project.jl"))
include(joinpath(@__DIR__, "potential_defs.jl"))

params = RoperParams()

# `channel_decomp=true` keeps Fock-channel weights of each requested state.
result = compute!(project, my_V, params;
                  backend=:projected_blocks,
                  channel_filter=channel_filter,
                  entry_filter=entry_filter,
                  channel_decomp=true)

output_file = joinpath(@__DIR__, "spectrum.tsv")
write_spectrum(result, output_file)


