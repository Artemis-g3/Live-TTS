from __future__ import annotations

import argparse

from role_data import discover_available_roles, prepare_role_excel


def main() -> None:
    parser = argparse.ArgumentParser(description="根据角色标定 CSV 生成角色专属情绪关键词 Excel。")
    parser.add_argument("--role", default="all", help="角色名，或使用 all 处理全部角色")
    parser.add_argument("--force", action="store_true", help="强制重建角色 Excel")
    args = parser.parse_args()

    if args.role == "all":
        roles = discover_available_roles()
    else:
        roles = [args.role.strip()]

    if not roles:
        raise RuntimeError("未发现可用角色，无法生成角色 Excel。")

    for role in roles:
        role_paths = prepare_role_excel(role, force=args.force)
        print(f"prepared_role={role_paths.role}")
        print(f"excel_path={role_paths.excel_path}")


if __name__ == "__main__":
    main()
