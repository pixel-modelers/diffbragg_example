import h5py
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys

# --- Configuration ---
# Number of ROI pairs (Data + Model) to show per page
ROI_PAIRS_PER_PAGE = 12
# Grid layout for the main image area (must multiply to ROI_PAIRS_PER_PAGE)
GRID_ROWS = 3
IMAGE_COLS = 4  # The total number of columns for the images is now 3
# TOTAL_FIG_COLS is no longer needed but would be 3

# Check if the grid layout is valid
if GRID_ROWS * IMAGE_COLS != ROI_PAIRS_PER_PAGE:
    print("Configuration Error: GRID_ROWS * IMAGE_COLS must equal ROI_PAIRS_PER_PAGE.", file=sys.stderr)
    sys.exit(1)


class HDF5Viewer:
    """
    Interactive Matplotlib viewer for diffraction data stored in an HDF5 file.
    Displays Data/Model pairs in a paginated grid.
    """

    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.data = self._load_data()

        if self.data is None:
            sys.exit(1)

        self.num_rois = len(self.data['data'])
        self.num_pages = int(np.ceil(self.num_rois / ROI_PAIRS_PER_PAGE))
        self.current_page = 0

        # Initialize the figure for interactive use
        plt.ion()
        # Adjusted figure size for a 3-column-only layout
        self.fig = plt.figure(figsize=(8,3.5))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        print(f"Loaded {self.num_rois} ROIs from '{self.hdf5_path}'.")
        print("Use RIGHT/LEFT arrow keys to navigate pages. Press 'q' to quit.")

        self._plot_page()

    def _load_data(self):
        """Loads and processes data from the HDF5 file."""
        try:
            with h5py.File(self.hdf5_path, 'r') as h:

                scores = h["score"][()]
                # Reconstruct the model image (mod_im = bg_scale*bg_im + bragg_scale*bragg_im + offset)
                print("Reconstructing model images...")
                bragg_im =[h["bragg"]["roi%d" % i][()] for i in range(len(scores))]
                bg_im =[h["bg"]["roi%d" % i][()] for i in range(len(scores))]
                model_im =[h["model"]["roi%d" % i][()] for i in range(len(scores))]
                #bragg_scale = h["bragg_scale"][()]
                #bg_scale = h["bg_scale"][()]
                #offset = h["offset"][()]

                #model_im = [(bg_s * bg) + (br_s * br) + o
                #              for bg_s, bg, br_s, br, o in zip(bg_scale, bg_im, bragg_scale, bragg_im, offset)]

                return {
                    'data': [h["data"]["roi%d" % i][()] for i in range(len(scores))],
                    'model': model_im,
                    'scores': scores,
                }
        except FileNotFoundError:
            print(f"Error: File not found at '{self.hdf5_path}'", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error loading HDF5 data: {e}", file=sys.stderr)
            return None

    def _plot_page(self):
        """Draws the current page of images."""
        plt.clf()

        start_idx = self.current_page * ROI_PAIRS_PER_PAGE
        end_idx = min(start_idx + ROI_PAIRS_PER_PAGE, self.num_rois)

        # --- Image Grid Setup ---
        # The grid now only contains the image columns (IMAGE_COLS)
        gs = self.fig.add_gridspec(GRID_ROWS, IMAGE_COLS, wspace=0.05, hspace=0.24)

        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01)

        for i in range(start_idx, end_idx):
            # i_page is the index on the current page (0 to ROI_PAIRS_PER_PAGE - 1)
            i_page = i - start_idx

            # Calculate grid position for GRID_ROWS rows x IMAGE_COLS columns of images
            row = i_page // IMAGE_COLS
            col = i_page % IMAGE_COLS

            # 1. Create the composite image [Data | Black Line | Model]
            data_im = self.data['data'][i]
            model_im = self.data['model'][i]

            # Create a 2-pixel wide black line (0 intensity) with the same height
            height, _ = data_im.shape
            black_line = np.zeros((height, 1)) * np.nan

            # Horizontally stack the images
            composite_im = np.hstack([data_im, black_line, model_im])

            # Plot the composite image in a single subplot
            ax = self.fig.add_subplot(gs[row, col])

            # Use the max value across both data and model for a consistent color scale (vmax)
            m = data_im.mean()
            s = data_im.std()
            vmin = m - s
            vmax = m + 3 * s
            m = model_im.mean()
            s = model_im.std()
            vmin = min(m - s, vmin)
            vmax = max(m + 3 * s, vmax)
            ax.imshow(composite_im, interpolation='none', cmap='cividis', vmax=vmax, vmin=vmin)

            # Set title and remove ticks
            ax.set_title(f"ROI {i} | Score: {self.data['scores'][i]*100:.1f}", fontsize=9, pad=0)
            ax.grid(1)
            y, x = data_im.shape
            z = x + black_line.shape[1] - 0.5
            ax.set_xticks([x//2, x - 0.5, z, z + x//2])
            ax.set_yticks([y//2])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)
            ax.grid(1, ls="--", lw=0.8, color="#777777", alpha=0.8)

        # Highlight current page in the title
        self.fig.suptitle(
            f"Diffraction ROI Fit Viewer: Page {self.current_page + 1} / {self.num_pages} "
            f"(ROIs {start_idx} - {end_idx - 1}) | Press 'q' to exit",
            fontsize=14,
            fontweight='bold'
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for suptitle
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _on_key_press(self, event):
        """Handles key presses for navigation."""
        if event.key in ('right', 'up'):
            if self.current_page < self.num_pages - 1:
                self.current_page += 1
                self._plot_page()
        elif event.key in ('left', 'down'):
            if self.current_page > 0:
                self.current_page -= 1
                self._plot_page()
        elif event.key == 'q':
            plt.close(self.fig)
            plt.ioff()

    def show(self):
        """Blocks execution until the plot window is closed."""
        plt.show(block=True)


if __name__ == "__main__":
    ap = ArgumentParser(description="Interactive Matplotlib viewer for HDF5 diffraction ROI data.")
    ap.add_argument(
        "hdf5_path",
        type=str,
        help="Path to the output HDF5 file containing 'data', 'bragg', 'bg', and 'score' datasets."
    )
    args = ap.parse_args()

    viewer = HDF5Viewer(args.hdf5_path)
    viewer.show()