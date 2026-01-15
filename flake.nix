{
  description = "Chrome Dino Game RL Agent";
  nixConfig.bash-prompt-prefix = "[nix(dino)] ";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
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
        xorg.libxcb
        rdma-core
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "dino-dev";

        packages =
          with pkgs;
          [
            # Python and package manager
            python314
            uv

            # Build dependencies for compiling Python packages
            linuxHeaders # Required by evdev (pynput dependency)
            rdma-core # Required by CUDA libraries (PyTorch dependency)
            pkg-config # Required by pycairo
            cairo # Required by pycairo

            # GStreamer and PipeWire for Wayland screen capture
            gst_all_1.gstreamer
            gst_all_1.gst-plugins-base
            gst_all_1.gst-plugins-good
            gst_all_1.gst-plugins-bad
            pipewire
            wireplumber

            # GObject introspection for PyGObject
            gobject-introspection
            gtk3

            # D-Bus for XDG Desktop Portal
            dbus
            dbus-glib

            # Qt and X11 libraries for OpenCV GUI (cv2.imshow, selectROI)
            libsForQt5.qt5.qtbase
            xorg.xcbutil
            xorg.xcbutilwm
            xorg.xcbutilimage
            xorg.xcbutilkeysyms
            xorg.xcbutilrenderutil

            tmux
          ]
          ++ runtimeLibs;

        shellHook = ''
          # Set up library paths for runtime dependencies
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH"

          # Critical: evdev searches for kernel headers in C_INCLUDE_PATH
          export C_INCLUDE_PATH="${pkgs.linuxHeaders}/include"

          # GObject introspection for PyGObject
          export GI_TYPELIB_PATH="${pkgs.gst_all_1.gstreamer.out}/lib/girepository-1.0:${pkgs.gst_all_1.gst-plugins-base}/lib/girepository-1.0:${pkgs.gtk3}/lib/girepository-1.0:${pkgs.gobject-introspection}/lib/girepository-1.0"

          # GStreamer plugin path
          export GST_PLUGIN_PATH="${
            pkgs.lib.makeSearchPath "lib/gstreamer-1.0" [
              pkgs.gst_all_1.gst-plugins-base
              pkgs.gst_all_1.gst-plugins-good
              pkgs.gst_all_1.gst-plugins-bad
            ]
          }"

          # Qt plugin path for OpenCV
          export QT_PLUGIN_PATH="${pkgs.libsForQt5.qt5.qtbase}/lib/qt-${pkgs.libsForQt5.qt5.qtbase.version}/plugins"
          export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGIN_PATH/platforms"
          export QT_QPA_PLATFORM=xcb

          # Add Qt library path so opencv-python's bundled Qt plugins can find dependencies
          export LD_LIBRARY_PATH="${pkgs.libsForQt5.qt5.qtbase}/lib:$LD_LIBRARY_PATH"

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
