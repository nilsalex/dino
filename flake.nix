{
  description = "Chrome Dino Game RL Agent";
  nixConfig.bash-prompt-prefix = "[nix(dino)] ";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      # Runtime libraries needed by Python packages
      runtimeLibs = with pkgs; [
        libGL
        glib
        stdenv.cc.cc.lib
        xorg.libX11
        xorg.libXext
        xorg.libXinerama
        xorg.libXtst
        xorg.libXrandr
        xorg.libXfixes
        rdma-core
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "dino-dev";

        packages = with pkgs; [
          # Python and package manager
          python314
          uv

          # Build dependencies for compiling Python packages
          linuxHeaders  # Required by evdev (pynput dependency)
          rdma-core     # Required by CUDA libraries (PyTorch dependency)
        ] ++ runtimeLibs;

        shellHook = ''
          # Set up library paths for runtime dependencies
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH"

          # Critical: evdev searches for kernel headers in C_INCLUDE_PATH
          export C_INCLUDE_PATH="${pkgs.linuxHeaders}/include"

          echo "Chrome Dino RL Agent development environment"
          echo "Python: $(python --version)"
          echo "uv: $(uv --version)"
          echo ""
          echo "Run 'uv sync' to install Python dependencies"
          echo "Run 'uv run python main.py' to start training"
        '';
      };
    };
}
