{
  description = "A reproducible Python development environment with PyTorch and Vision models.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable"; 
    flake-utils.url = "github:numtide/flake-utils"; 
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { 
          inherit system; 
          config = {
            allowUnfree = true; 
          };
        }; 
	      python = pkgs.python313;

        # List of Python packages to install
        pythonPackages = with python.pkgs; [
          # Deep Learning Framework
          torch
          torchvision

          # Vision Transformer Models
          transformers

          # Data Processing
          numpy
          pandas
          pydicom
          pillow

          # Experiment Tracking
          wandb
          python-dotenv

          # Metrics and Evaluation
          scikit-learn
          scipy

          # Visualization
          matplotlib
          seaborn
          plotly

          # Configuration
          pyyaml

          # Progress Bars
          tqdm

          # Jupyter
          jupyter
          ipykernel
          ipywidgets

          # Utilities
          glob2
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.cudatoolkit
            python
          ] ++ pythonPackages;

          shellHook = ''
            export CUDA_PATH=${pkgs.cudatoolkit}
            export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${pkgs.cudatoolkit}/lib:$LD_LIBRARY_PATH
          '';
        };
      }
    );
}
