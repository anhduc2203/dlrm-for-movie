from flask import Flask
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

@api.resource("/<string:handler>")
class RecommendResource(Resource):
    def get(self, handler):
        parser = reqparse.RequestParser()
        parser.add_argument("guid", type=str, default=None)
        parser.add_argument("itemid", type=str, default=None)

        args = parser.parse_args()
        if handler == 'recommend':
            result, alg = recommend(args["guid"], args["itemid"])
            return self.format_result(result, alg), 200
        # if handler == 'debug':
        #     result, alg = app_manager.recommend_manager.recommend(args["guid"], args["itemid"], args["alg"], True)
        #     return self.format_result(result, alg, True), 200
        return "Unknown Handler [{}]".format(handler), 400

    @staticmethod
    def format_result(result, alg, is_debug=False):
        payload = []
        #title_manager = app_manager.item_manager.support_info_managers[app_manager.constant.TITLE_MANAGER_ID]
        for item_id, score in sorted(result.items(), key=lambda kv: kv[1], reverse=True):
            item = {"id": item_id}
            if is_debug:
                item["score"] = str(score)
                #item["title"] = title_manager.get_by_item_id(item_id)
            payload.append(item)
        print("Response through restful...............")
        return {
            "recommend": payload,
            "algid": alg
        }

if __name__=="__main__":
    app.run(debug=False, host='0.0.0.0', port=2203)
