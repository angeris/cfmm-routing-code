{
  description = "CFFM routing";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = inputs@{ flake-parts, poetry2nix, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }:
        {
          devShells.default = pkgs.mkShell {
            LD_LIBRARY_PATH = with pkgs; lib.strings.makeLibraryPath [
              stdenv.cc.cc.lib
              zlib
              zlib.dev
              zlib.out
            ];

            nativeBuildInputs = with pkgs; [
              poetry
              python3
              zlib
              zlib.dev
              pkg-config
              (texliveSmall.withPackages
                (ps: with ps; [ gensymb ]))
            ];
            enterShell = ''
              poetry lock --no-update
              poetry instal --no-root
            '';
          };
        };
    };
}
