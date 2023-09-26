# Design Space Exploration of 3D Concrete Printed Bridges

![frontpage](https://github.com/DennisHollanders/Design-Space-Exploration-of-3D-Concrete-Printed-Bridges/assets/67959863/7d495cf2-d7ec-4d46-8e57-3f79c426f202)


This repository contains the source code for a 3D concrete printing optimization tool, which combines Finite Element Analysis (FEA) and generative design techniques. Additionally, it includes a user-guided dashboard for navigating the design space. This tool is designed to aid engineers and designers in the early design stage for 3D-printed concrete structures. The link to thesis can be found below 

#### Link to thesis: <to be updated> 

## Content 
- Thesis Abstract 
- Features
- Installation and usage

## Thesis Abstract 
3D concrete printing is rapidly emerging and numerous projects have showcased the potential of this technique. However, realizing the full scope of 3D printing properties in design applications remains challenging. This thesis therefore addresses the challenges of 3D concrete printing in design applications. To bridge this gap, a user-guided Design Space Exploration (DSE) approach is proposed. The focus lies on optimizing bridge designs, a suitable domain due to its alignment with the capabilities of 3D-printed concrete. The proposed dashboard aids users in identifying desirable topologies (i.e. “What to design?”) and guides them in decision-making by illustrating parametric implications within the design space (i.e. “How to design”).

The general methodology comprises two steps: the creation of a parametric bridge and the development of the dashboard, used to navigate the design space. A generative design strategy is used to unify these steps, traversing from a single bridge to a design space for dashboard analysis. To elaborate, the parameters making up a single bridge, shape- and optimization parameters, will be cross-referenced. Design principles are established to ensure that the created bridges are print-specific. They include two-dimensional design, modular, sizing optimization, unreinforced concrete, and diversity of topologies. These are used in the parametric development of a bridge. Which starts with varying design domains, making up the line model or ground structure of each bridge, yielding diverse topologies in the generative design process. The next steps in the process include assigning structural properties to the line model, after which a structural calculation will determine the strain energies in the system. Dividing the strain energy by the total strain energy gives the strain energy ratio of each element to that of the system. This ratio is the driver behind the heuristic optimization that follows. In the further development of the optimization algorithm, tools have been added to develop its functionality. One of which is the Young’s modulus penalization, which tricks the optimization into optimizing towards compressive structures by fictively reducing the Young's modulus of elements in tension. Additionally, a tool is created able to set a construction constraint to the optimization, forcing material to a location by setting a minimum strain energy bound in the optimization process. Finally, Numerical examples are used to benchmark or exemplify the tools making up the design space of a single bridge.
This design space resulting from the described process, is being navigated with a dashboard, created in dash, a Plotly library. The dashboard equips users with an array of tools, enabling them to identify bridges that align with their preferences. This involves defining self-set criteria within the design space, allowing users to make informed decisions through analysis graphs. The dashboard uses a total of six graphs to comply with this functionality: a domain constraint graph, filtering histogram, 3D objective space, t-SNE plot, parallel categories plot and a visualization tool. To ensure the ability for a comparative analysis, the inserted databases are being ‘normalized’ via a rank transformation, which assigns a rank to each element based on its specific output variable. 
To validate the effectiveness of the dashboard, a design case has been formulated. The scenario concerns a bridge with an eight-meter span, in which the variable domain heights are confined to a range of positive and negative three.
The results from the case showed that the “What to design?” could be answered effectively. Regardless of the complexity of the design question, the dashboard could provide a comprehension comparison between optima designs. The index selection tool facilitates a straightforward comparison between identified bridge designs and desired outcomes. Notably, the ranking method eliminates the complication in normalization due to outliers. However, it does introduce a trade-off by diminishing the ease of comparing due to the absence of relative differences. The ability to compare how different design queries lead to different optima is perceived to be a functional method to convey decision-making with non-technical personnel. Secondly the dashboard should provide information about the design space. Through the utilization of the parallel categories plot, information could be gained regarding the distribution of input variables and their impact on output performance. The results have shown that this provided basic information about domain functioning, showing that low values for domain 3 perform poor in general. 
Studying the parameters has shown that prestress seemed to be a positive addition to the structures’ functioning. The Young's modulus penalization tools behave very case-specific. While one penalization method occasionally outperforms the others, a slight preference is observed for non-penalized structures. The Construction constraint tool causes lower total strain energy performances, as expected, but does however proof not to be a postprocessing step; it distinctly alters the distribution of material around the constrained element. It is therefore seen as a valuable tool to control material placement throughout the optimization. In addition to this, the generative design process in combination with optimization is seen as a valuable methodology to impose significant variation to the topological design space.

To conclude, the dashboard has shown to be able to address both research objectives, providing information on the “what and How to design?”. While the methodology applied showed to provide bridges tailored to the specific needs of 3D concrete printed bridges. Consequently, this underscores the potential of user-guided design space exploration as an effective tool for engineers and designers, navigating the complex design landscape of 3D concrete printed structures.


## Features
- Grasshopper file (FEA/Optimization code implemented)
- FEA/Optimization code in seperate py file
- Code for the dashboard:
      - Main code for initializing and importing relevant data to the dashboard.
      - CallbacksLayout: Dash code containing the layout of the dashboard and callbackfunctions making up the graphs.
- Dataset with 5832 bridge variants
      - The standard settings of the grasshopper file contain the parametric values used, making up this design space. 

## Installation and usage
#### installation of RhinoWIP and setting environments in RhinoCode
At the time of writing 02-10-23 RhinoWIP was used for its new RhinoCode functionality allowing for the implementation of Python3 in the grasshopper environment
- Download version of Rhino with RhinoCode (RhinoWIP: https://www.rhino3d.com/download/rhino-for-windows/beta )
- Open a grasshopper file
- Set the environments:
    Use # requirements: <package-specifier>
        # r: <package-specifier == version>
  Check: https://discourse.mcneel.com/t/rhino-8-feature-scripteditor-cpython-csharp/128353/30 for more info (Pandas libraries seemed to bug out one way of installing them would be to directly copy the relevant library into the add-ons for rhinocode by diving into the folder structure. Another way would be to run: ,before importing the pandas library
The requirements file contains the used versions of each package which can also be used as version specifier for the RhinoCode environment.

#### Usage of the grasshopper file 
- Open the Grasshopper file for used in this thesis; You should be able to use to grasshopper code by now allowing you to generate bridge databases.
- You are able to use the generative framework in which you control which bridges are being generated by altering the panel input. Check the amount of variants this creates then set the maximum slider value to this (This slider is found on the blue background). Enable the storage, set the slider to 0, reset the storage and animate the slider (use amount of frames = amount of options) to generate a database with the variants applied by the panel input.
The generative framework crossrefences the panel input options to create all available options between the presented parametric inputs. This results in lists with indices of parameter settings of which each index corresponds to a single bridge. (columns are the parameter options, rows are the parameters of a single bridge). By now indexing over this lists, you obtain the parameters making up each bridge. Therefore animating the slider used for indexing results generating all available bridges.

- The framework below the generative framework is the 'manual' framework. This framework allows you to explore individual variants, implying that you have to set each parameter and a single bridge results. The grasshopper script behind the 'manual' framework is the same as that of the generative framework only now single options are presented to the cross-referencing tool resulting in a single bridge as indexing option.
  
#### Running the dash application 
- Set the correct folder directy for the generated bridges database.
- Check the manual inputs these variables are in full capital letters. 
- Run the Main_dashboard application.
