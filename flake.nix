{
  description = "Personal aggregator devShell (M0 skeleton; packages + module land in M4)";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let pkgs = nixpkgs.legacyPackages.${system}; in {
      devShells.default = pkgs.mkShell {
        packages = [
          pkgs.python311
          pkgs.uv
          pkgs.ruff
          pkgs.sqlite
          pkgs.gitleaks
          pkgs.gh
        ];
      };
    });
}
