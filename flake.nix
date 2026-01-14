{
  description = "python env";
  nixConfig.bash-prompt-prefix = "[nix(python)] ";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        name = "python env";
        buildInputs = [
          pkgs.python314
          pkgs.uv
        ];
      };
    };
}
