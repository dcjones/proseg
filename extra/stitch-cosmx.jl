#!/usr/bin/env julia
#
# Compute global coordinates for each transcript.

using Glob: glob
using CSV
using DataFrames
using CodecZlib
using ArgParse

argparser = ArgParseSettings()
@add_arg_table! argparser begin
    "path"
        help = "Directory for raw CosMx files"
        required = true
    "output"
        help = "Output transcripts csv file"
        required = true
end


function main()
    args = parse_args(argparser)
    path = args["path"]
    if !ispath(path)
        error("$(path) is not a valid path")
    end

    output_filename = args["output"]

    config_filename = glob("S0/*/RunSummary/*_ExptConfig.txt", path)[1]
    println("Config filename: ", config_filename)
    width, height, pixel_size = read_image_size(config_filename)
    println("Image size: $(width)μm × $(height)μm")
    println("Pixel size: $(pixel_size)μm")

    fov_filename = glob("S0/*/RunSummary/latest.fovs.csv", path)[1]
    println("FOV filename: ", fov_filename)
    fovs = read_fov_positions(fov_filename)
    println("Coordinates found for $(length(fovs)) FOVs")

    fov_paths = glob("S0/*/AnalysisResults/*/FOV*", path)
    println("Transcript info found for $(length(fov_paths)) FOVs")

    println("Processing FOVs...")
    output = GzipCompressorStream(open(output_filename, "w"))
    for (i, fov_path) in enumerate(fov_paths)
        fov = parse(Int, match(r"FOV(\d+)$", fov_path).captures[1])
        println("  fov $(fov)")
        df = compute_global_positions(fov_path, fov, pixel_size, fovs)
        CSV.write(output, df, append=i > 1)
    end
    close(output)
end


function read_image_size(config_filename::String)
    config_text = String(read(open(config_filename)))
    height = parse(Float64, match(r"Image Rows: (\d+)", config_text).captures[1])
    width = parse(Float64, match(r"Image Cols: (\d+)", config_text).captures[1])

    pixel_size = parse(Float64, match(r"ImPixel_nm: (\d+)", config_text).captures[1])
    pixel_size /= 1e3 # convert nm to μm

    width *= pixel_size
    height *= pixel_size

    return (width, height, pixel_size)
end

function read_fov_positions(fov_filename::String)
    fovs_df = CSV.read(fov_filename, DataFrame, header=false)
    col_names = ["slide", "x", "y", "z", "zoffset", "ROI", "FOV"]
    if size(fovs_df, 2) == 8
        push!(col_names, "acqOrder")
    end
    rename!(fovs_df, col_names)
    fovs = Dict{Int, Tuple{Float64, Float64, Float64}}()
    for fov in eachrow(fovs_df)
        fovs[fov.FOV] = (fov.x * 1e3, fov.y * 1e3, fov.z * 1e3)
    end
    return fovs
end

function compute_global_positions(fov_path::String, fov::Int, pixel_size::Float64, fovs::Dict)
    filename = glob("*_complete_code_cell_target_call_coord.csv", fov_path)[1]
    df = CSV.read(filename, DataFrame)

    df = df[:,[:fov, :CellId, :x, :y, :z, :target, :CellComp]]
    rename!(df, [:fov, :cell_ID, :x_local_px, :y_local_px, :z, :target, :CellComp])
    @assert all(df.fov .== fov)

    # FOV offsets
    x_fov, y_fov, z_fov = fovs[fov]

    x = df.x_local_px * pixel_size .- x_fov
    y = .-df.y_local_px * pixel_size .- y_fov

    insertcols!(df, :x => x, :y => y)

    return df
end


main()

