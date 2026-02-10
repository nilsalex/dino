{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-23-05.url = "github:NixOS/nixpkgs/nixos-23.05";
    nixpkgs-24-05.url = "github:NixOS/nixpkgs/nixos-24.05";
  };

  outputs =
    {
      nixpkgs,
      nixpkgs-23-05,
      nixpkgs-24-05,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pkgs-23-05 = nixpkgs-23-05.legacyPackages.${system};
          pkgs-24-05 = nixpkgs-24-05.legacyPackages.${system};

          # Playwright dependencies (Firefox browser libraries)
          playwrightPkgs = with pkgs; [
            glib-networking
            libnghttp2
            libpsl
            libdrm
            libmanette
            wayland
            hyphen
            libtasn1
            libsecret
            enchant
            fontconfig
            freetype
            libpng
            libjpeg8
            libepoxy
            libavif
            libwebp
            flite
            libgpg-error
            libgcrypt
            libopus
            libevent
            pkgs-24-05.libvpx
            woff2
            lcms
            libxslt
            sqlite
            pkgs-23-05.libxml2
            zlib
            xz
            libgcc.lib
            icu74
            graphene
            vulkan-loader
            gdk-pixbuf
            harfbuzzFull
            harfbuzz
            gtk3
            gtk4
            pango
            cups
            nspr
            nss
            at-spi2-atk
            expat
            libX11
            libXcomposite
            libXdamage
            libXext
            libXfixes
            libXrandr
            libxcb
            libxkbcommon
            libgbm
            systemd
            alsa-lib
          ];
        in
        {
          default = pkgs.mkShell {
            packages = [
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
            ]
            ++ playwrightPkgs;

            env = lib.optionalAttrs pkgs.stdenv.isLinux {
              LD_LIBRARY_PATH = lib.makeLibraryPath (
                pkgs.pythonManylinuxPackages.manylinux1
                ++ [
                  pkgs.gst_all_1.gstreamer
                  pkgs.cairo
                ]
                ++ playwrightPkgs
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
              PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS = "true";
              GIO_MODULE_DIR = "${pkgs.glib-networking}/lib/gio/modules";
            };

            shellHook = ''
              export PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright";
              unset PYTHONPATH
              uv sync
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
