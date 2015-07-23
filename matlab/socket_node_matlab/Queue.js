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
