{
  description = "DLLM Project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          pip
          numpy
          flask
          matplotlib
        ]);
        systemPackages = with pkgs; [
          bun
          python-launcher
        ];
        dllm_proj = pkgs.stdenv.mkDerivation {
          pname = "dllm-project";
          name = "dllm-project";
          buildInputs = [ pythonEnv ] ++ systemPackages;
          src=self;
          installPhase = ''
            mkdir -p $out/bin
            cat > $out/bin/dllm-project << EOF
            #!/usr/bin/env bash
            cd ./source/client
            ${pkgs.bun}/bin/bun install
            ${pkgs.bun}/bin/bun build client.js --outdir ../assets/

            cd ../../
            ${pythonEnv}/bin/python source/main.py
            EOF
            chmod +x $out/bin/dllm-project
          '';
        };
      in rec {
        devShell = pkgs.mkShell {
          buildInputs = [ pythonEnv ] ++ systemPackages;
        };
        packages.default = dllm_proj;

        apps.default = {
          type = "app";
          drv = dllm_proj;
          program = "${dllm_proj}/bin/dllm-project";
        };
      });
}
