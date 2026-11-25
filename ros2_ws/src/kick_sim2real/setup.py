from setuptools import setup

package_name = "kick_sim2real"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/kick_sim2real.yaml"]),
        ("share/" + package_name + "/launch", ["launch/kick_sim2real.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="unsigned123",
    maintainer_email="cathing@126.com",
    description="Sim-to-real deployment for the hierarchical_kick upper policy",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy_node = kick_sim2real.node:main",
            "export_policy = kick_sim2real.export:main",
        ],
    },
)
