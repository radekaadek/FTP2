import click
import open3d as o3d
import numpy as np

@click.command()
@click.argument(
    'input_pcd',
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    '--normal-radius',
    type=click.FLOAT,
    default=0.1,
    show_default=True,
    help='Radius for KDTree search during normal estimation.'
)
@click.option(
    '--normal-max-nn',
    type=click.INT,
    default=30,
    show_default=True,
    help='Maximum neighbors for KDTree search during normal estimation.'
)
@click.option(
    '--poisson',
    is_flag=True,
    default=False,
    help='Perform Poisson surface reconstruction.'
)
@click.option(
    '--poisson-depth',
    type=click.INT,
    default=9,
    show_default=True,
    help='Depth parameter for Poisson reconstruction.'
)
@click.option(
    '--crop/--no-crop',
    default=True,
    show_default=True,
    help='Crop the Poisson mesh to the bounding box of the input cloud.'
)
@click.option(
    '--poisson-output',
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help='Save the resulting Poisson mesh to this file path (e.g., poisson_mesh.ply).'
)
@click.option(
    '--ball-pivoting',
    is_flag=True,
    default=False,
    help='Perform Ball Pivoting surface reconstruction.'
)
@click.option(
    '--ball-pivoting-output',
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help='Save the resulting Ball Pivoting mesh to this file path (e.g., bp_mesh.stl).'
)
@click.option(
    '--visualize/--no-visualize',
    default=True,
    show_default=True,
    help='Visualize the generated mesh(es) in an Open3D window.'
)
def process_cloud(input_pcd, normal_radius, normal_max_nn,
                   poisson, poisson_depth, crop, poisson_output,
                   ball_pivoting, ball_pivoting_output,
                   visualize):
    """
    Processes a point cloud file (INPUT_PCD) using Open3D.

    Estimates normals and optionally reconstructs the surface using
    Poisson Surface Reconstruction and/or Ball Pivoting algorithms.
    The resulting meshes can be visualized and/or saved to files.
    """
    bp_radii_mult = [1.5, 2.0, 2.5, 3.0]

    if not poisson and not ball_pivoting and not poisson_output and not ball_pivoting_output:
        click.echo("Warning: Neither Poisson (--poisson) nor Ball Pivoting (--ball-pivoting) "
                   "reconstruction requested, and no output files specified. "
                   "The script will only load the cloud and estimate normals.", err=True)

    click.echo(f"Loading point cloud from: {input_pcd}")
    try:
        cloud = o3d.io.read_point_cloud(input_pcd)
        if not cloud.has_points():
            click.echo(f"Error: Point cloud file {input_pcd} is empty or could not be read correctly.", err=True)
            return 1
    except Exception as e:
        click.echo(f"Error loading point cloud: {e}", err=True)
        return 1

    click.echo("Estimating normals...")
    try:
        cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
        cloud.normalize_normals()
        click.echo(f"Normals estimated (radius={normal_radius}, max_nn={normal_max_nn}) and normalized.")
    except Exception as e:
        click.echo(f"Error estimating normals: {e}", err=True)

    if not cloud.has_normals():
         click.echo("Warning: Normals could not be estimated successfully. Reconstruction might fail or produce poor results.", err=True)

    geometries_to_visualize = []
    poisson_mesh_result = None
    ball_pivoting_mesh_result = None

    if poisson or poisson_output:
        click.echo(f"Performing Poisson reconstruction (depth={poisson_depth})...")
        try:
            poisson_mesh_result, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=poisson_depth)
            if not poisson_mesh_result or not poisson_mesh_result.has_triangles():
                 click.echo("Warning: Poisson reconstruction resulted in an empty or invalid mesh.", err=True)
                 poisson_mesh_result = None
            else:
                click.echo("Poisson reconstruction successful.")
                if crop:
                    click.echo("Cropping Poisson mesh to input cloud bounds...")
                    bbox = cloud.get_axis_aligned_bounding_box()
                    poisson_mesh_result = poisson_mesh_result.crop(bbox)
                    click.echo("Cropping complete.")

                if poisson_output:
                    click.echo(f"Saving Poisson mesh to: {poisson_output}")
                    if not o3d.io.write_triangle_mesh(poisson_output, poisson_mesh_result, write_ascii=False, compressed=True):
                         click.echo(f"Error: Failed to save Poisson mesh to {poisson_output}", err=True)
                    else:
                         click.echo("Poisson mesh saved.")

                if visualize and poisson_mesh_result:
                     geometries_to_visualize.append(poisson_mesh_result)

        except Exception as e:
            click.echo(f"Error during Poisson reconstruction: {e}", err=True)
            poisson_mesh_result = None

    if ball_pivoting or ball_pivoting_output:
        click.echo("Calculating average point distance for Ball Pivoting...")
        try:
            dists = cloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(dists)
            click.echo(f"Average distance between points: {avg_dist:.4f}")

            radii_values = [avg_dist * m for m in bp_radii_mult]
            if not radii_values:
                click.echo("Error: No radii specified or calculated for Ball Pivoting.", err=True)
            else:
                click.echo(f"Using Ball Pivoting radii: {radii_values}")
                ball_pivoting_mesh_result = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    cloud,
                    o3d.utility.DoubleVector(radii_values)
                )

                if not ball_pivoting_mesh_result or not ball_pivoting_mesh_result.has_triangles():
                     click.echo("Warning: Ball Pivoting reconstruction resulted in an empty or invalid mesh.", err=True)
                     ball_pivoting_mesh_result = None
                else:
                    click.echo("Ball Pivoting reconstruction successful.")

                    if ball_pivoting_output:
                        click.echo(f"Saving Ball Pivoting mesh to: {ball_pivoting_output}")
                        if not o3d.io.write_triangle_mesh(ball_pivoting_output, ball_pivoting_mesh_result, write_ascii=False, compressed=True):
                             click.echo(f"Error: Failed to save Ball Pivoting mesh to {ball_pivoting_output}", err=True)
                        else:
                             click.echo("Ball Pivoting mesh saved.")

                    if visualize and ball_pivoting_mesh_result:
                        geometries_to_visualize.append(ball_pivoting_mesh_result)

        except Exception as e:
            click.echo(f"Error during Ball Pivoting setup or reconstruction: {e}", err=True)
            ball_pivoting_mesh_result = None

    if visualize and geometries_to_visualize:
        click.echo(f"Visualizing {len(geometries_to_visualize)} generated mesh(es)...")
        try:
            o3d.visualization.draw_geometries(geometries_to_visualize, window_name="Open3D Mesh Reconstruction")
        except Exception as e:
            click.echo(f"Error during visualization: {e}", err=True)
    elif visualize:
        click.echo("Visualization requested, but no valid meshes were generated or selected for visualization.")

    click.echo("Processing finished.")
    return 0


if __name__ == "__main__":
    process_cloud()
