
predictions = [
    {
        'question': "What is the primary purpose of the ALOM plugin?",
        'answer': "The ALOM plugin integrates OmniSwitch management into the Milestone VMS,allowing users to control port and switch functions directly from the Milestone interface. This eliminates the need to switch between applications.",
        'result': "The primary purpose of the ALOM plugin is to provide information and control over various components of the Milestone VMS (Video Management System) architecture, such as the OmniSwitch, its ports, and connected cameras. This includes details like port status, PoE power usage, camera names, IP addresses, and more. It also allows operators to perform actions like resetting cameras or checking their connectivity without needing to access a separate user interface."
    }
]

import deepeval_provider
deep_eval_model = deepeval_provider.make_model(model="llama3.1")
correctness_metric = deepeval_provider.run_correctness_eval(deep_eval_model, predictions[0])