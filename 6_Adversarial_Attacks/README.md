
EVASION - Adversarial ML
This task is about performing an evasion attack. For this you have to create so-called "adversarial examples" that look like normal images but deceive the target model.
The flow of the unit is as follows: you can ask a local service for a challenge and they will give you a list of indices from the training dataset and a list of labels to predict. The training dataset is known and available to you. Your task is now to manipulate the images in such a way that they

    a) by the model in the service as the default
       target class are predicted and
    b) hardly differ from the original image
       (have a low norm of the difference between both images).
To solve the challenge, upload the manipulated images to the local service and you will receive an encrypted token in response. This can be entered on the web platform to try to log in permanently. Only then does it become apparent how well the challenge has been solved. The score is calculated from the above criteria.
The service can be reached via http://127.0.0.1:8000 and provides the following REST API for the various interactions:

    - Unit activation (GET): /api/activate?token=s0m3t0k3n
    - Query prediction (POST): /evasion/api/predict
    - create new challenge (GET): /evasion/api/get_challenge
    - Solve Challenge (POST): /evasion/api/solve_challenge
In the training data is a README file that specifies the details for the service's POST requests and responses. The local service token is the same as that used to activate this unit and only needs to be entered once. The other functions of the service are then activated.

Have fun and happy evasion!