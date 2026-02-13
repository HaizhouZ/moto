import moto
import time
import json
import argparse

class benchmark_tool:
    def __init__(self, arg_parser: argparse.ArgumentParser = argparse.ArgumentParser()):
        self.stats = []
        self.__arg_parser = arg_parser
        self.__arg_parser.add_argument("--output_file", type=str)
        self.__arg_parser.add_argument("--config", type=str)
        self.__config = None
        self.__args = None

    @property
    def config(self):
        if self.__config is None:
            with open(self.__args.config, "r") as f:
                self.__config = json.load(f)
        return self.__config

    @property
    def arg_parser(self):
        return self.__arg_parser
    
    @property
    def args(self):
        if self.__args is None:
            self.__args = self.arg_parser.parse_args()
        return self.__args

    def run(self, sqp: moto.sqp, label: str, config_idx: int, config_detail):
        except_caught = False
        start = time.perf_counter()
        try:
            info = sqp.update(100, verbose=False)
        except Exception as e:
            except_caught = True
            print(
                f"Exception during SQP solve for case {label} config {config_idx}: {e}"
            )
        end = time.perf_counter()

        tim = end - start

        self.stats.append(
            {
                "label": label,
                "config": config_idx,
                "config_detail": config_detail,
                "solved": info.solved,
                "num_iter": info.num_iter,
                "num_qp_iter": info.num_iter,
                "time": tim,
                "failure_mode": {
                    "prim_infeas": info.inf_prim_res > sqp.settings.prim_tol,
                    "dual_infeas": info.inf_dual_res > sqp.settings.dual_tol,
                    "comp_infeas": info.inf_comp_res > sqp.settings.comp_tol,
                    "nan_inf": except_caught,
                },
            }
        )

    def dump(self):
        with open(self.__args.output_file, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"Stats dumped to {self.__args.output_file}")