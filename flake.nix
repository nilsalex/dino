{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.xvfb-run
              pkgs.opencode
              pkgs.python312
              pkgs.uv
              pkgs.gst_all_1.gstreamer
              pkgs.gst_all_1.gst-plugins-base
              pkgs.gst_all_1.gst-plugins-good
              pkgs.gst_all_1.gst-plugins-bad
              pkgs.cairo
              pkgs.pkg-config
              pkgs.dbus
              pkgs.poppler-utils
            ];

            env = lib.optionalAttrs pkgs.stdenv.isLinux {
              LD_LIBRARY_PATH = lib.makeLibraryPath (
                pkgs.pythonManylinuxPackages.manylinux1
                ++ [
                  pkgs.gst_all_1.gstreamer
                  pkgs.cairo
                ]
              );
              GI_TYPELIB_PATH = lib.makeSearchPath "lib/girepository-1.0" [
                pkgs.gst_all_1.gstreamer.out
                pkgs.gst_all_1.gst-plugins-base
              ];
              UV_PYTHON = pkgs.python312;
              PKG_CONFIG_PATH = lib.makeSearchPath "lib/pkgconfig" [
                pkgs.cairo
                pkgs.dbus
              ];
              NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
              GIO_MODULE_DIR = "${pkgs.glib-networking}/lib/gio/modules";
            };

            shellHook = ''
              unset PYTHONPATH
              uv sync
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
