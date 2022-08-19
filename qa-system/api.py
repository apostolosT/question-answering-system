import logging
from flask import Blueprint, jsonify, request
from pyparsing import traceback
from src.predict import predict_answer


api = Blueprint('api', __name__)


@api.route('/prediction', methods=['POST'])
def predict():
    """
    file: swagger/predict.yaml
    """
    request_body = request.get_json()
    print(request_body)
    context = request_body['context']
    question = request_body['question']

    prediction = predict_answer(context, question)

    return jsonify(prediction), 200


# @api.errorhandler(Exception)
# def handle_error(e):
#     logging.error(str(e), exc_info=1)
#     if isinstance(e, export_errors()):
#         if hasattr(e, 'status_code'):
#             e.stack = traceback.format_exc()
#             response = e.to_dict()
#             response.status_code = e.status_code
#             return response
#         return jsonify({
#             'error': {
#                 'statusCode': 400,
#                 'name': type(e).__name__,
#                 'message': repr(e),
#                 'stack': traceback.format_exc()
#             }
#         })

#     raise e
