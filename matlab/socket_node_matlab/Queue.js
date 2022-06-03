/*
 * Queue.js
 * Copyright (c) 2016 Xianglan Piao <lanxlpiao@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * @file Queue.js
 * @author Xianglan Piao <lanxlpiao@gmail.com>
 * Date: 2016.10.07
 */
function Queue() {
    a = [];
	this.getLength = function() {
	    return a.length
	}

	this.isEmpty = function() {
	    return 0 == a.length
	}

	this.enqueue = function(b) {
	    a.push(b)
	}

	this.dequeue = function() {
	    a.shift();
	}

	this.toString = function() {
	    text = '';
	    for (i = 0; i < a.length; i++) { 
	        text += a[i] + '\n';
	    }
	    console.log(text);
	}
}

module.exports = Queue;
